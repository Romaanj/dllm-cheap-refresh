"""
Phase-14a (Dream-7B) — End-to-end inference port from LLaDA to Dream-7B.

Same algorithmic axes as phase14a_cheap_e2e.py but adapted to Dream's
naming conventions:
  - block.attn_norm      → layer.input_layernorm
  - block.q/k/v_proj     → layer.self_attn.q/k/v_proj
  - block.attn_out       → layer.self_attn.o_proj
  - block.ff_norm        → layer.post_attention_layernorm
  - block.ff_proj/up_proj → layer.mlp.gate_proj/up_proj
  - block.ff_out         → layer.mlp.down_proj
  - block.rotary_emb     → layer.self_attn.rotary_emb (different API)

MASK_ID = 151666 (Dream-v0-Instruct-7B)

Usage:
  CUDA_VISIBLE_DEVICES=0 python phase14a_dream.py --mode baseline \
    --num-samples 4 --out-dir results_phase14a_dream/baseline_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/inverse_cdf/Fast-dLLM/dream")
from model.modeling_dream import DreamModel, rotate_half  # type: ignore
from model.configuration_dream import DreamConfig  # type: ignore

DREAM_MASK_ID = 151666


# -------------------- Sparse Dream layer forward --------------------

def apply_rope_one(x, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to a single tensor (B, H, M, hd) given cos, sin (B, M, hd)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def sparse_dream_layer_forward_with_kv(
    layer, hybrid_input, A_l, A_input_changed, prev_block_out, k_cache, v_cache,
):
    """Sparse forward for one Dream decoder layer with incremental KV cache.

    - Sparse Q proj at A_l only + RoPE at A_l positions
    - K, V update at A_input_changed only (cached otherwise)
    - SDPA on (|A|, L)
    - o_proj on |A|
    - Sparse FFN on |A| rows
    - Scatter into prev_block_out (clone)
    """
    B, L, C = hybrid_input.shape
    attn = layer.self_attn
    n_q = attn.num_heads
    n_kv = attn.num_key_value_heads
    hd = attn.head_dim

    x_normed = layer.input_layernorm(hybrid_input)

    # SLOT 2: incremental K/V projection at A_input_changed positions
    if A_input_changed is not None and A_input_changed.numel() > 0:
        x_partial = x_normed.index_select(1, A_input_changed)
        k_partial = attn.k_proj(x_partial)
        v_partial = attn.v_proj(x_partial)
        k_cache.index_copy_(1, A_input_changed, k_partial)
        v_cache.index_copy_(1, A_input_changed, v_partial)

    # Sparse Q proj at A_l rows
    x_A = x_normed.index_select(1, A_l)
    q_A_raw = attn.q_proj(x_A)
    n_A = q_A_raw.shape[1]
    q_A = q_A_raw.view(B, n_A, n_q, hd).transpose(1, 2)        # (B, n_q, n_A, hd)
    k = k_cache.view(B, L, n_kv, hd).transpose(1, 2)            # (B, n_kv, L, hd)
    v = v_cache.view(B, L, n_kv, hd).transpose(1, 2)

    # RoPE — Q at A_l positions, K at all L positions
    pos_A = A_l.unsqueeze(0)                                     # (1, n_A)
    cos_A, sin_A = attn.rotary_emb(q_A, pos_A)                   # (1, n_A, hd)
    q_A = apply_rope_one(q_A, cos_A, sin_A)
    pos_full = torch.arange(L, device=k.device, dtype=torch.long).unsqueeze(0)
    cos_K, sin_K = attn.rotary_emb(k, pos_full)
    k = apply_rope_one(k, cos_K, sin_K)

    # GQA expand
    if n_q != n_kv:
        k_exp = k.repeat_interleave(n_q // n_kv, dim=1)
        v_exp = v.repeat_interleave(n_q // n_kv, dim=1)
    else:
        k_exp, v_exp = k, v

    att = F.scaled_dot_product_attention(q_A, k_exp, v_exp, attn_mask=None, dropout_p=0.0)
    att = att.transpose(1, 2).contiguous().view(B, n_A, C)
    att = attn.o_proj(att)                                       # (B, n_A, d)

    # Residual at A_l only, then sparse FFN
    y_attn_A = hybrid_input.index_select(1, A_l) + att
    h2 = layer.post_attention_layernorm(y_attn_A)
    ffn_out_A = layer.mlp(h2)
    y_out_A = y_attn_A + ffn_out_A

    # Scatter into prev cache
    new_block_out = prev_block_out.clone()
    new_block_out[:, A_l, :] = y_out_A
    return new_block_out


def init_dream_kv_cache(layers, n_layers, block_in_per_layer):
    """Compute initial K, V cache for each Dream layer from given inputs."""
    k_cache, v_cache = [], []
    for l in range(n_layers):
        with torch.inference_mode():
            x_normed = layers[l].input_layernorm(block_in_per_layer[l])
            k_cache.append(layers[l].self_attn.k_proj(x_normed).clone())
            v_cache.append(layers[l].self_attn.v_proj(x_normed).clone())
    return k_cache, v_cache


def sparse_dream_hybrid_cascade(
    layers, n_layers, cur_block_in_0, prev_block_out_per_layer,
    A_per_layer, k_cache, v_cache, U_t,
):
    hybrid = cur_block_in_0.clone()
    A_input_changed = U_t
    new_block_outs = []
    for l in range(n_layers):
        with torch.inference_mode():
            new_out = sparse_dream_layer_forward_with_kv(
                layers[l], hybrid, A_per_layer[l], A_input_changed,
                prev_block_out_per_layer[l], k_cache[l], v_cache[l],
            )
        new_block_outs.append(new_out.clone())
        hybrid = new_out
        A_input_changed = A_per_layer[l]
    return hybrid, new_block_outs


# -------------------- Capture for Dream layers --------------------

class DreamAllLayerCapture:
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.block_in = {}
        self.block_out = {}
        self._handles = []

    def install(self, model):
        for li, layer in enumerate(model.model.layers):
            def make_pre(lid):
                def hk(module, args):
                    self.block_in[lid] = args[0].detach()
                    return None
                return hk
            def make_post(lid):
                def hk(module, args, output):
                    out = output[0] if isinstance(output, tuple) else output
                    self.block_out[lid] = out.detach()
                    return None
                return hk
            self._handles.append(layer.register_forward_pre_hook(make_pre(li)))
            self._handles.append(layer.register_forward_hook(make_post(li)))

    def snapshot_block_in_0(self):
        return self.block_in[0].clone()

    def snapshot_block_out_all(self):
        return [self.block_out[l].clone() for l in range(self.n_layers)]


# -------------------- E3 estimator helper --------------------

def window_set(U, w, T, device):
    if U.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    offs = torch.arange(-w, w + 1, device=device).view(1, -1)
    grid = (U.view(-1, 1) + offs).reshape(-1).clamp(0, T - 1)
    return torch.unique(grid)


def build_E3_A(lag1_sorted_layer, U_t, K, T, device, w=4):
    """Original Python-set version (kept for parity comparison)."""
    W = window_set(U_t, w, T, device)
    W_set = set(W.cpu().tolist())
    out = list(W.cpu().tolist())[:K]
    if len(out) < K:
        for p in lag1_sorted_layer.cpu().tolist():
            if p in W_set:
                continue
            out.append(p)
            if len(out) >= K:
                break
    return torch.tensor(out[:K], dtype=torch.long, device=device)


def build_W_and_mask(U_t, w, T, device):
    """Precompute W = U ∪ window(U,w) once per step. Returns (W tensor, W_mask bool)."""
    W = window_set(U_t, w, T, device)
    W_mask = torch.zeros(T, dtype=torch.bool, device=device)
    if W.numel() > 0:
        W_mask[W] = True
    return W, W_mask


def build_E3_A_gpu(lag1_sorted_layer, W, W_mask, K, T):
    """GPU-native E3 active-set builder.

    Args:
      lag1_sorted_layer: (T,) long, positions sorted by lag-1 norm desc
      W: (|W|,) long, precomputed U_t ∪ window(U_t, w) positions
      W_mask: (T,) bool, True at positions inside W
      K: int target size
      T: int seq length

    Returns:
      A: (K,) long — W positions first, then top-(K-|W|) lag-1 positions excluding W
    """
    n_W = W.numel()
    if n_W >= K:
        return W[:K]
    n_extra = K - n_W
    # Boolean mask: True at sorted positions NOT in W
    not_in_W = ~W_mask[lag1_sorted_layer]              # (T,) bool
    # First n_extra True indices (stable, in lag-1 rank order)
    extra_sort_idx = torch.nonzero(not_in_W, as_tuple=True)[0][:n_extra]
    extra_positions = lag1_sorted_layer[extra_sort_idx]
    return torch.cat([W, extra_positions], dim=0)


# --- inlined helpers (avoid LLaDA model package conflict) ---

def load_gsm8k(seed, split):
    return load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=seed)


def format_gsm8k_demo(sample):
    return f"Question: {sample['question']}\nAnswer: {sample['answer']}"


def build_gsm8k_fewshot_text(sample, demo_ds, num_shots):
    demos = []
    if num_shots > 0:
        for demo in demo_ds:
            demos.append(format_gsm8k_demo(demo))
            if len(demos) >= num_shots:
                break
    target = f"Question: {sample['question']}\nAnswer:"
    return "\n\n".join(demos + [target]) if demos else str(sample["question"])


def get_prompt_text(tokenizer, sample, demo_ds, num_shots, no_chat_template):
    text = build_gsm8k_fewshot_text(sample, demo_ds, num_shots)
    if no_chat_template:
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, tokenize=False,
    )


def get_num_transfer_tokens(block_mask_index, steps):
    device = block_mask_index.device
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    n = base.unsqueeze(1).expand(-1, steps).to(torch.long)
    cols = torch.arange(steps, device=device).unsqueeze(0)
    return n + (cols < rem.unsqueeze(1)).to(torch.long)


# -------------------- Dream-aware utils --------------------

def pick_transfer_greedy(logits, x, block_mask, k, mask_id):
    """Top-k by confidence (greedy)."""
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


def pick_transfer_threshold(logits, x, block_mask, threshold):
    """Parallel decoding: unmask all positions with conf >= threshold + force max."""
    p = F.softmax(logits.to(torch.float32), dim=-1)
    x0 = logits.argmax(dim=-1)
    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    conf = torch.where(block_mask, x0_p, neg_inf)
    transfer = block_mask & (conf >= threshold)
    max_idx = torch.argmax(conf, dim=1, keepdim=True)
    force = torch.zeros_like(transfer).scatter_(1, max_idx, True)
    transfer = (transfer | force) & block_mask
    return transfer, x0


# -------------------- GSM8K answer parsing --------------------

def extract_pred_answer(text: str) -> Optional[str]:
    m = re.search(r"####\s*([\-\d\.,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip().rstrip(".")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def extract_gold_answer(answer_field: str) -> Optional[str]:
    m = re.search(r"####\s*([\-\d\.,]+)", answer_field)
    if m:
        return m.group(1).replace(",", "").strip().rstrip(".")
    return None


def normalize_num(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip().rstrip(".")
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return f"{f:g}"
    except ValueError:
        return s


# -------------------- Inference loop --------------------

def run_sample(model, tokenizer, capture, sample, demo_ds, args, device):
    prompt_text = get_prompt_text(tokenizer, sample, demo_ds, args.num_fewshot, no_chat_template=False)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc.input_ids.to(device)
    prompt_len = int(input_ids.shape[1])
    total_len = prompt_len + args.gen_length

    x = torch.full((1, total_len), DREAM_MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    num_blocks = args.gen_length // args.block_length
    steps_per_block = args.steps // num_blocks
    use_parallel = args.threshold > 0.0
    use_cheap_mode = args.mode == "cheap_e3"
    global_step = 0

    n_layers = len(model.model.layers)
    layers = model.model.layers
    norm_f = model.model.norm
    lm_head = model.lm_head

    prev_snap = None
    prev_prev_snap = None
    prev_transfer = None
    k_cache = None
    v_cache = None
    last_warmup_block_in_per_layer = None

    t0 = time.perf_counter()
    n_cheap = 0
    n_base = 0
    for nb in range(num_blocks):
        block_start = prompt_len + nb * args.block_length
        block_end = prompt_len + (nb + 1) * args.block_length
        block_mask_init = (x[:, block_start:block_end] == DREAM_MASK_ID)
        num_transfer = get_num_transfer_tokens(block_mask_init, steps_per_block)

        for i in range(steps_per_block):
            cur_mask = (x == DREAM_MASK_ID)
            block_mask = cur_mask.clone()
            block_mask[:, :block_start] = False
            block_mask[:, block_end:] = False
            if block_mask.sum() == 0:
                break
            if not use_parallel:
                k_step = int(num_transfer[0, i].item())
                if k_step <= 0:
                    break

            T_len = int(x.shape[1])
            do_cheap_this_step = (
                use_cheap_mode and global_step >= args.warmup
                and prev_snap is not None and prev_prev_snap is not None and prev_transfer is not None
            )

            if not do_cheap_this_step:
                with torch.inference_mode():
                    out = model(x)
                # Dream next-token shift
                logits_step = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
                new_prev = capture.snapshot_block_out_all() if capture is not None else None
                n_base += 1
                if capture is not None and use_cheap_mode:
                    cur_block_in_0 = capture.snapshot_block_in_0()
                    last_warmup_block_in_per_layer = [cur_block_in_0] + new_prev[:-1]
            else:
                embed = model.model.embed_tokens
                with torch.inference_mode():
                    cur_block_in_0 = embed(x).detach()
                # Batched lag-1 ranking: stack all layers, single norm + argsort
                prev_stack = torch.stack(prev_snap, dim=0)          # (L, 1, T, D) bf16
                prev_prev_stack = torch.stack(prev_prev_snap, dim=0)
                d1_all = (prev_stack.to(torch.float32)
                          - prev_prev_stack.to(torch.float32)).norm(dim=-1)[:, 0, :]  # (L, T)
                lag1_sorted_all = torch.argsort(d1_all, dim=-1, descending=True)       # (L, T)
                lag1_sorted = list(lag1_sorted_all)  # list of (T,) views for downstream API
                U_t = prev_transfer[0].nonzero(as_tuple=True)[0]
                # K-schedule
                if args.k_schedule == "uniform":
                    K_per_layer = [args.K] * n_layers
                elif args.k_schedule == "stepped":
                    third = n_layers // 3
                    K_per_layer = ([args.K // 2] * third + [args.K] * third
                                   + [int(args.K * 0.75)] * (n_layers - 2 * third))
                else:
                    raise ValueError(args.k_schedule)
                # Hoist W + W_mask out of per-layer loop (same for all layers in this step)
                W_step, W_mask_step = build_W_and_mask(U_t, args.window, T_len, device)
                A_per_layer = []
                for l, K_l in enumerate(K_per_layer):
                    A_per_layer.append(build_E3_A_gpu(lag1_sorted[l], W_step, W_mask_step, min(K_l, T_len), T_len))

                if args.no_kv_cache:
                    # Oracle-style hybrid forward: full Q/K/V projection at every layer
                    # (matches phase12 oracle mechanism — isolates KV-cache stale issue)
                    from phase12_cascade_oracle_dream import dream_hybrid_cascade_forward
                    position_ids_full = torch.arange(T_len, device=device, dtype=torch.long).unsqueeze(0)
                    with torch.inference_mode():
                        position_embeddings = model.model.rotary_emb(cur_block_in_0, position_ids_full)
                    h_final, new_prev = dream_hybrid_cascade_forward(
                        layers, n_layers, cur_block_in_0, prev_snap, A_per_layer,
                        position_embeddings, position_ids_full,
                        return_mixed_per_layer=True,
                    )
                else:
                    if k_cache is None:
                        k_cache, v_cache = init_dream_kv_cache(layers, n_layers, last_warmup_block_in_per_layer)
                    h_final, new_prev = sparse_dream_hybrid_cascade(
                        layers, n_layers, cur_block_in_0, prev_snap, A_per_layer, k_cache, v_cache, U_t,
                    )
                with torch.inference_mode():
                    h_norm = norm_f(h_final)
                    logits_full = lm_head(h_norm)
                    # Dream next-token shift
                    logits_step = torch.cat([logits_full[:, :1], logits_full[:, :-1]], dim=1)
                n_cheap += 1

            if use_parallel:
                transfer, x0 = pick_transfer_threshold(logits_step, x, block_mask, args.threshold)
            else:
                transfer, x0 = pick_transfer_greedy(logits_step, x, block_mask, k_step, DREAM_MASK_ID)
            x = torch.where(transfer, x0, x)
            if use_cheap_mode:
                prev_prev_snap = prev_snap
                prev_snap = new_prev
                prev_transfer = transfer.clone()
            global_step += 1
            if (x[:, block_start:block_end] == DREAM_MASK_ID).sum() == 0:
                break

    wall = time.perf_counter() - t0

    gen_ids = x[0, prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    gold = normalize_num(extract_gold_answer(sample.get("answer", "")))
    pred = normalize_num(extract_pred_answer(gen_text))
    correct = (gold is not None and pred is not None and gold == pred)
    return {"gen_text": gen_text, "gold": gold, "pred": pred, "correct": correct,
            "wall_s": wall, "n_steps": global_step, "n_cheap": n_cheap, "n_base": n_base}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="test")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--start-sample", type=int, default=0)
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--mode", choices=["baseline", "cheap_e3"], default="baseline")
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--k-schedule", default="uniform")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--no-kv-cache", action="store_true",
                   help="Disable KV cache; full Q/K/V projection every step (Phase 12 oracle mechanism). Slower but isolates KV-cache stale-state from algorithm.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[phase14a-dream] loading {args.model}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DreamConfig.from_pretrained(args.model, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.model, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[phase14a-dream] model loaded, n_layers={config.num_hidden_layers}", flush=True)

    # Install hooks for cheap mode
    capture = None
    if args.mode == "cheap_e3":
        capture = DreamAllLayerCapture(config.num_hidden_layers)
        capture.install(model)

    test_ds = load_gsm8k(args.seed, args.split)
    demo_ds = load_gsm8k(args.seed, "train") if args.num_fewshot > 0 else None

    out_jsonl = os.path.join(args.out_dir, "results.jsonl")
    f_out = open(out_jsonl, "w")
    n_correct = 0
    total_wall = 0.0
    for idx in range(args.start_sample, args.start_sample + args.num_samples):
        sample = test_ds[idx]
        r = run_sample(model, tokenizer, capture, sample, demo_ds, args, device)
        row = {"sample_id": idx, "mode": args.mode, "threshold": args.threshold, **r}
        f_out.write(json.dumps(row) + "\n"); f_out.flush()
        if r["correct"]:
            n_correct += 1
        total_wall += r["wall_s"]
        status = "✓" if r["correct"] else "✗"
        print(f"[phase14a-dream] sample {idx:>4} {status}  pred={r['pred']} gold={r['gold']}  "
              f"{r['wall_s']:.1f}s  steps={r['n_steps']}  cheap/base={r['n_cheap']}/{r['n_base']}", flush=True)
    f_out.close()
    acc = n_correct / max(args.num_samples, 1)
    print(f"[phase14a-dream] DONE: {n_correct}/{args.num_samples} = {acc:.3f}  total {total_wall:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"n": args.num_samples, "n_correct": n_correct, "accuracy": acc, "total_wall_s": total_wall}, f, indent=2)


if __name__ == "__main__":
    main()
