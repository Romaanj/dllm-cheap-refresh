"""
Phase-9 Day 3 — Composed end-to-end Δblock_output quality (Path A).

For each deep layer at each step transition k → k+1, we compute three forms
of Δblock_output (block output diff) and compare them:

  1. Δblock_TRUE      = block_out_{k+1, baseline} − block_out_k_baseline
                       (= what the natural full forward produces)

  2. Δblock_ANALYTIC_EXACT_FFN
                      = Δy_attn_analytic + [FFN(y_attn_k + Δy_attn_analytic) − FFN(y_attn_k)]
                      (= attention via Sherman-Morrison + EXACT FFN forward on the
                        analytic Δy_attn perturbation; assumes attn substitute is correct,
                        full FFN cost retained)

  3. Δblock_ANALYTIC_LINEAR_FFN
                      = Δy_attn_analytic + J_FFN(y_attn_k) · Δy_attn_analytic
                      (= production form: linearized FFN, no full forward;
                        J · v computed via fp32 finite-difference)

Where
  Δy_attn_analytic = attn_out(o_analytic) − att_k        (cat_b col substituted via Sherman-Morrison)
  (for non-cat_b positions; x_input cached so Δx_input = 0)

Metrics per (layer, step) on non-cat_b positions:
  cos(2 vs 1)  cos(3 vs 1)  cos(3 vs 2)
  relmse        norms        magnitudes

Output JSONL.

Usage:
  CUDA_VISIBLE_DEVICES=0 python phase9_day3_composed.py \
      --num-samples 2 --gen-length 256 --layers 15,20,25,28,31 \
      --eps 1e-3 --out-dir results_phase9_day3
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from gsm8k_attention_sink_drift_eval import (
    get_prompt_text,
    load_gsm8k,
)
from model.modeling_llada import LLaDAModelLM


MASK_ID = 126336


# ---------------------------------------------------------------------------
# Capture: attention internals (Q,K,V,α,Z,o,att_post_proj) + block in/out
# ---------------------------------------------------------------------------


class FullCapture:
    def __init__(self, layers: List[int]):
        self.layers = list(layers)
        self.attn_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.block_in: Dict[int, torch.Tensor] = {}
        self.block_out: Dict[int, torch.Tensor] = {}
        self.att_post_proj: Dict[int, torch.Tensor] = {}  # what attention() returns (after attn_out)
        self.y_attn: Dict[int, torch.Tensor] = {}        # x + att (input to ff_norm)

    def install(self, model: Any):
        blocks = model.model.transformer.blocks
        layer_set = set(int(li) for li in self.layers)

        # block pre/post hooks
        for li, block in enumerate(blocks):
            if li not in layer_set:
                continue

            def make_pre(layer_id):
                def hook(module, args):
                    self.block_in[layer_id] = args[0].detach().clone()
                    return None
                return hook
            def make_post(layer_id):
                def hook(module, args, output):
                    out_tensor = output[0] if isinstance(output, tuple) else output
                    self.block_out[layer_id] = out_tensor.detach().clone()
                    return None
                return hook
            block.register_forward_pre_hook(make_pre(li))
            block.register_forward_hook(make_post(li))

            # ff_norm input = y_attn = block_input + dropout(att_post_proj)
            def make_yattn(layer_id):
                def hook(module, args):
                    self.y_attn[layer_id] = args[0].detach().clone()
                    return None
                return hook
            block.ff_norm.register_forward_pre_hook(make_yattn(li))

        # Monkey-patch attention to expose internals (and post-proj att)
        capture = self

        def install_attention_capture():
            for li, block in enumerate(blocks):
                if li not in layer_set:
                    continue
                original_attention = block.attention

                def make_wrapped(orig, layer_id, block_ref):
                    def wrapped(q, k, v, attention_bias=None,
                                layer_past=None, use_cache=False,
                                replace_position=None, output_attentions=False):
                        # Call original to perform forward (returns att_post_proj already
                        # since attn_out is applied inside attention method)
                        att, cache_kv, attn_weights = orig(
                            q, k, v, attention_bias=attention_bias,
                            layer_past=layer_past, use_cache=use_cache,
                            replace_position=replace_position, output_attentions=True,
                        )
                        capture.att_post_proj[layer_id] = att.detach().clone()

                        # Replicate internals: post-RoPE Q, K, V, scores, softmax, o per-head
                        B, T, C = q.size()
                        cfg = orig.__self__.config
                        n_heads = cfg.n_heads
                        n_kv = cfg.effective_n_kv_heads
                        head_dim = C // n_heads
                        q_r = q.view(B, T, n_heads, head_dim).transpose(1, 2)
                        k_r = k.view(B, T, n_kv, head_dim).transpose(1, 2)
                        v_r = v.view(B, T, n_kv, head_dim).transpose(1, 2)
                        if cfg.rope:
                            q_post, k_post = orig.__self__.rotary_emb(q_r, k_r)
                        else:
                            q_post, k_post = q_r, k_r
                        if n_heads != n_kv:
                            assert n_heads % n_kv == 0
                            k_exp = k_post.repeat_interleave(n_heads // n_kv, dim=1)
                            v_exp = v_r.repeat_interleave(n_heads // n_kv, dim=1)
                        else:
                            k_exp = k_post
                            v_exp = v_r
                        scale = 1.0 / math.sqrt(head_dim)
                        q32 = q_post.to(torch.float32)
                        k32 = k_exp.to(torch.float32)
                        v32 = v_exp.to(torch.float32)
                        s = (q32 @ k32.transpose(-2, -1)) * scale
                        s_max = s.max(dim=-1, keepdim=True).values
                        exp_s = (s - s_max).exp()
                        Z = exp_s.sum(dim=-1, keepdim=False)
                        alpha = exp_s / Z.unsqueeze(-1)
                        o_ph = alpha @ v32

                        # Drop full (B, n_q, T, T) tensors (s, alpha, exp_s); keep small derivatives
                        capture.attn_state[layer_id] = {
                            "q_post": q_post.detach(),
                            "k_exp": k_exp.detach(),
                            "v_exp": v_exp.detach(),
                            "s_max": s_max.detach(),
                            "Z": Z.detach(),
                            "o_ph": o_ph.detach(),
                        }
                        # Explicit free of large temporaries
                        del s, exp_s, alpha
                        return att, cache_kv, attn_weights
                    return wrapped

                block.attention = make_wrapped(original_attention, li, block)

        install_attention_capture()

    def snapshot_layer(self, li: int) -> Optional[Dict[str, torch.Tensor]]:
        if li not in self.attn_state:
            return None
        snap = {
            "block_in": self.block_in.get(li).clone() if self.block_in.get(li) is not None else None,
            "block_out": self.block_out.get(li).clone() if self.block_out.get(li) is not None else None,
            "att_post_proj": self.att_post_proj.get(li).clone() if self.att_post_proj.get(li) is not None else None,
            "y_attn": self.y_attn.get(li).clone() if self.y_attn.get(li) is not None else None,
            "attn": {k: v.clone() for k, v in self.attn_state[li].items()},
        }
        return snap

    def clear(self):
        self.attn_state.clear()
        self.block_in.clear()
        self.block_out.clear()
        self.att_post_proj.clear()
        self.y_attn.clear()


# ---------------------------------------------------------------------------
# fp32 helpers
# ---------------------------------------------------------------------------


def attn_out_fp32(block, o_concat_fp32: torch.Tensor) -> torch.Tensor:
    """Apply block.attn_out in fp32 by casting weight on the fly."""
    return F.linear(
        o_concat_fp32,
        block.attn_out.weight.to(torch.float32),
        block.attn_out.bias.to(torch.float32) if block.attn_out.bias is not None else None,
    )


def run_ffn_only_fp32(block: Any, y_attn_fp32: torch.Tensor) -> torch.Tensor:
    fn = block.ff_norm
    eps = getattr(fn, "eps", 1e-5)
    if hasattr(fn, "weight") and fn.weight is not None:
        w = fn.weight.to(torch.float32)
    else:
        w = None
    rms = y_attn_fp32.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps).rsqrt()
    x = y_attn_fp32 * rms
    if w is not None:
        x = x * w
    x1 = F.linear(x, block.ff_proj.weight.to(torch.float32),
                  block.ff_proj.bias.to(torch.float32) if block.ff_proj.bias is not None else None)
    x2 = F.linear(x, block.up_proj.weight.to(torch.float32),
                  block.up_proj.bias.to(torch.float32) if block.up_proj.bias is not None else None)
    x1 = F.silu(x1)
    z = x1 * x2
    out = F.linear(z, block.ff_out.weight.to(torch.float32),
                   block.ff_out.bias.to(torch.float32) if block.ff_out.bias is not None else None)
    return out


# ---------------------------------------------------------------------------
# Sherman-Morrison o_analytic_new (per-head, pre-attn_out, in fp32)
# ---------------------------------------------------------------------------


def sherman_morrison_o(cache_old: Dict[str, torch.Tensor],
                       cache_new: Dict[str, torch.Tensor],
                       cat_b_idx: torch.Tensor) -> torch.Tensor:
    Q = cache_old["q_post"]
    K_old = cache_old["k_exp"]
    V_old = cache_old["v_exp"]
    s_max_old = cache_old["s_max"]
    Z_old = cache_old["Z"]
    o_old = cache_old["o_ph"]
    K_new = cache_new["k_exp"]
    V_new = cache_new["v_exp"]
    cat_b_idx = cat_b_idx.to(K_old.device).long()
    scale = 1.0 / math.sqrt(K_old.size(-1))
    K_cb_new = K_new[:, :, cat_b_idx, :]
    V_cb_new = V_new[:, :, cat_b_idx, :]
    K_cb_old = K_old[:, :, cat_b_idx, :]
    V_cb_old = V_old[:, :, cat_b_idx, :]
    # Recompute s at cat_b columns on demand (avoid storing full s tensor)
    s_new_at_catb = (Q @ K_cb_new.transpose(-2, -1)) * scale         # (B, n_q, T, n_catb)
    s_old_at_catb = (Q @ K_cb_old.transpose(-2, -1)) * scale         # (B, n_q, T, n_catb)
    exp_old = (s_old_at_catb - s_max_old).exp()
    exp_new = (s_new_at_catb - s_max_old).exp()
    delta_Z = (exp_new - exp_old).sum(dim=-1)
    Z_new = Z_old + delta_Z
    contrib_new = torch.einsum("bhpc,bhcd->bhpd", exp_new, V_cb_new)
    contrib_old = torch.einsum("bhpc,bhcd->bhpd", exp_old, V_cb_old)
    o_new_unnorm = Z_old.unsqueeze(-1) * o_old - contrib_old + contrib_new
    o_new = o_new_unnorm / Z_new.unsqueeze(-1).clamp_min(1e-30)
    return o_new  # (B, n_q, T, head_dim)


# ---------------------------------------------------------------------------
# Decoding loop helpers (1-token-per-step low_confidence)
# ---------------------------------------------------------------------------


def get_num_transfer_tokens(block_mask_index, steps):
    device = block_mask_index.device
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    n = base.unsqueeze(1).expand(-1, steps).to(torch.long)
    cols = torch.arange(steps, device=device).unsqueeze(0)
    return n + (cols < rem.unsqueeze(1)).to(torch.long)


def pick_transfer(logits, x, block_mask, k):
    p = F.softmax(logits.to(torch.float32), dim=-1)
    x0 = logits.argmax(dim=-1)
    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    conf = torch.where(block_mask, x0_p, neg_inf)
    transfer = torch.zeros_like(block_mask)
    if k <= 0:
        return transfer, x0
    _, idx = torch.topk(conf, k=k, dim=-1)
    transfer.scatter_(-1, idx, True)
    transfer = transfer & block_mask
    return transfer, x0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--layers", type=str, default="15,20,25,28,31")
    p.add_argument("--eps", type=float, default=1e-3,
                   help="finite-diff scale for J_FFN · v in linearized variant")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--start-sample", type=int, default=0)
    args = p.parse_args()

    layer_list = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[phase9-day3] loading model ...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLaDAModelLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    n_layers = model.config.n_layers
    blocks = model.model.transformer.blocks

    test_ds = load_gsm8k(args.seed, args.split)
    demo_ds = load_gsm8k(args.seed, "train") if args.num_fewshot > 0 else None

    capture = FullCapture(layer_list)
    capture.install(model)

    out_jsonl = os.path.join(args.out_dir, "day3_composed.jsonl")
    f_out = open(out_jsonl, "w", encoding="utf-8")

    for sample_id in range(args.start_sample, args.start_sample + args.num_samples):
        sample = test_ds[sample_id]
        prompt_text = get_prompt_text(tokenizer, sample, demo_ds, args.num_fewshot, no_chat_template=False)
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc.input_ids.to(device)
        prompt_len = input_ids.shape[1]
        total_len = prompt_len + args.gen_length
        x = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids

        num_blocks = args.gen_length // args.block_length
        steps_per_block = args.steps // num_blocks

        prev_snapshots: Dict[int, Dict[str, Any]] = {}
        prev_x_mask: Optional[torch.Tensor] = None

        global_step = 0
        for nb in range(num_blocks):
            block_start = prompt_len + nb * args.block_length
            block_end = prompt_len + (nb + 1) * args.block_length
            block_mask_init = (x[:, block_start:block_end] == MASK_ID)
            num_transfer = get_num_transfer_tokens(block_mask_init, steps_per_block)

            for i in range(steps_per_block):
                cur_mask = (x == MASK_ID)
                block_mask = cur_mask.clone()
                block_mask[:, :block_start] = False
                block_mask[:, block_end:] = False
                k_step = int(num_transfer[0, i].item())
                if k_step <= 0 or block_mask.sum() == 0:
                    break

                with torch.inference_mode():
                    out = model(x)
                logits = out.logits

                # Snapshot per layer
                cur_snaps = {li: capture.snapshot_layer(li) for li in layer_list}

                if prev_snapshots and prev_x_mask is not None:
                    just_unmasked = prev_x_mask & (~cur_mask)
                    cat_b_idx = just_unmasked[0].nonzero(as_tuple=False).flatten()
                    if cat_b_idx.numel() > 0:
                        for li in layer_list:
                            old = prev_snapshots.get(li)
                            new = cur_snaps.get(li)
                            if old is None or new is None:
                                continue
                            with torch.inference_mode():
                                block = blocks[li]
                                # ---- Sherman-Morrison o_analytic ----
                                o_analytic = sherman_morrison_o(old["attn"], new["attn"], cat_b_idx)  # (B, n_q, T, d_head)
                                # Concat heads: (B, T, n_q * d_head)
                                B, n_q, T, d_h = o_analytic.shape
                                o_concat_fp32 = o_analytic.transpose(1, 2).contiguous().view(B, T, n_q * d_h)
                                att_analytic_fp32 = attn_out_fp32(block, o_concat_fp32)         # (B, T, D)

                                att_old_fp32 = old["att_post_proj"].to(torch.float32)
                                d_att_analytic = att_analytic_fp32 - att_old_fp32              # for non-cat_b correct
                                # For cat_b queries we don't trust SM (their Q is fresh too). Mask out cat_b later.

                                y_old_fp32 = old["y_attn"].to(torch.float32)
                                # Δy_attn = Δblock_input (= cascade from upstream layers) + Δatt
                                # Day-3 isolates per-layer quality: use baseline Δblock_input
                                # (Day-4 will cascade analytic Δblock_input instead.)
                                d_block_input = (new["block_in"].to(torch.float32)
                                                 - old["block_in"].to(torch.float32))
                                d_y_attn = d_block_input + d_att_analytic
                                y_perturb = y_old_fp32 + d_y_attn

                                ffn_old = run_ffn_only_fp32(block, y_old_fp32)
                                ffn_perturb_exact = run_ffn_only_fp32(block, y_perturb)
                                d_ffn_exact = ffn_perturb_exact - ffn_old

                                # Linear approx: J · Δy via finite diff
                                eps = args.eps
                                ffn_perturb_eps = run_ffn_only_fp32(block, y_old_fp32 + eps * d_y_attn)
                                d_ffn_linear = (ffn_perturb_eps - ffn_old) / eps

                                # Δblock variants (residual: block_out = y_attn + FFN(y_attn))
                                # Δblock = Δy_attn + ΔFFN
                                d_block_exact = d_y_attn + d_ffn_exact          # analytic Δattn + exact FFN
                                d_block_linear = d_y_attn + d_ffn_linear        # analytic Δattn + linear FFN
                                d_block_true = (new["block_out"].to(torch.float32)
                                                - old["block_out"].to(torch.float32))

                                # Restrict to non-cat_b for the metric
                                mask_nc = torch.ones(T, dtype=torch.bool, device=d_block_true.device)
                                mask_nc[cat_b_idx] = False
                                a_exact = d_block_exact[0, mask_nc, :].reshape(-1)
                                a_lin = d_block_linear[0, mask_nc, :].reshape(-1)
                                t = d_block_true[0, mask_nc, :].reshape(-1)

                                cos_exact = F.cosine_similarity(a_exact.unsqueeze(0), t.unsqueeze(0)).item()
                                cos_lin = F.cosine_similarity(a_lin.unsqueeze(0), t.unsqueeze(0)).item()
                                cos_lin_vs_exact = F.cosine_similarity(a_lin.unsqueeze(0), a_exact.unsqueeze(0)).item()
                                rmse_exact = ((a_exact - t).pow(2).sum() / t.pow(2).sum().clamp_min(1e-30)).item()
                                rmse_lin = ((a_lin - t).pow(2).sum() / t.pow(2).sum().clamp_min(1e-30)).item()

                                row = {
                                    "sample_id": int(sample_id),
                                    "global_step": int(global_step),
                                    "layer": int(li),
                                    "n_cat_b": int(cat_b_idx.numel()),
                                    "cos_dblock_exact_vs_true": float(cos_exact),
                                    "cos_dblock_linear_vs_true": float(cos_lin),
                                    "cos_dblock_linear_vs_exact": float(cos_lin_vs_exact),
                                    "relmse_exact": float(rmse_exact),
                                    "relmse_linear": float(rmse_lin),
                                    "norm_dblock_true": float(t.norm().item()),
                                    "norm_dblock_exact": float(a_exact.norm().item()),
                                    "norm_dblock_linear": float(a_lin.norm().item()),
                                    "norm_dy_attn": float(d_y_attn[0, mask_nc, :].reshape(-1).norm().item()),
                                }
                                f_out.write(json.dumps(row) + "\n")
                                f_out.flush()

                transfer, x0 = pick_transfer(logits, x, block_mask, k_step)
                x = torch.where(transfer, x0, x)
                prev_snapshots = cur_snaps
                prev_x_mask = cur_mask
                global_step += 1
                if (x[:, block_start:block_end] == MASK_ID).sum() == 0:
                    break

        print(f"[phase9-day3] sample {sample_id} done", flush=True)

    f_out.close()
    print("[phase9-day3] DONE", flush=True)


if __name__ == "__main__":
    main()
