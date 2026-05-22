"""
Phase-14a — End-to-end inference with cheap E3 estimator on GSM8K.

For each step:
  - Steps 0..warmup-1: full baseline forward, save block_out_all (warm-up)
  - Steps warmup+: build A_l = U_t ∪ window(U_t, w) ∪ top-(K-...) by lag-1,
                   run hybrid cascade forward (refresh A_l, reuse prev for rest),
                   use hybrid logits for pick_transfer.

Self-bootstraps prev_snap from hybrid outputs after warm-up (no baseline
forward during cheap inference). Output: generated text + GSM8K accuracy.

Modes:
  --mode baseline   : always full baseline forward (reference)
  --mode cheap_e3   : E3 estimator after warmup

Usage:
  CUDA_VISIBLE_DEVICES=0 python phase14a_cheap_e2e.py \\
    --mode cheap_e3 --K 384 --warmup 2 \\
    --num-samples 4 --start-sample 0 \\
    --out-dir results_phase14a_e2e/cheap_gpu0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from gsm8k_attention_sink_drift_eval import (
    get_prompt_text,
    load_gsm8k,
)
from model.modeling_llada import LLaDAModelLM
from phase9_day3_composed import (
    get_num_transfer_tokens,
    pick_transfer,
    MASK_ID,
)
from sparse_block import (
    sparse_hybrid_cascade_forward,
    sparse_hybrid_cascade_forward_with_kv,
    init_kv_cache_from_block_inputs,
    masked_attention_fallback,
    masked_attention_sdpa,
    masked_attention_sdpa_sparse_q,
)


def pick_transfer_threshold(logits, x, block_mask, threshold):
    """Fast-dLLM-style parallel decoding: unmask all positions with conf >= threshold,
    plus the single max-confidence position (force progress).
    Returns (transfer_index, x0) like pick_transfer."""
    p = torch.softmax(logits.to(torch.float32), dim=-1)
    x0 = logits.argmax(dim=-1)
    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    conf = torch.where(block_mask, x0_p, neg_inf)
    transfer = block_mask & (conf >= threshold)
    # Force at least one (max-confidence) position to progress
    max_idx = torch.argmax(conf, dim=1, keepdim=True)
    force = torch.zeros_like(transfer).scatter_(1, max_idx, True)
    transfer = (transfer | force) & block_mask
    return transfer, x0


def pick_transfer_threshold_sparse_logits(
    h_final: torch.Tensor,             # (1, T, d) — hidden after all layers, before ln_f
    x: torch.Tensor,                    # (1, T) long — current sequence
    block_mask: torch.Tensor,           # (1, T) bool — positions valid for unmasking
    ln_f,                               # nn.Module — model.transformer.ln_f
    ff_out,                             # nn.Linear — output projection
    scale_logits: bool,
    d_model: int,
    threshold: float,
):
    """Equivalent to (ln_f + ff_out + pick_transfer_threshold) but ONLY computes
    ln_f and ff_out at block_mask positions. For LLaDA (vocab=126464, d=4096),
    ff_out is the most expensive single op; restricting it to ~32 positions
    instead of ~T saves a large fraction of per-step wall time, especially when
    the prompt is long (e.g., GSM8K T~1750).

    Mathematically equivalent under the assumption that pick_transfer_threshold
    only uses logits at block_mask positions (it does — the rest is masked to
    -inf right after argmax/softmax).

    Returns (transfer, x0) of full shape (1, T):
      - transfer: (1, T) bool — True where the position should be unmasked
      - x0:       (1, T) long — argmax token at unmasked positions (default = x)
    """
    T = int(x.shape[1])
    device = x.device
    idx = block_mask[0].nonzero(as_tuple=True)[0]   # (n_rel,) long
    if idx.numel() == 0:
        return torch.zeros((1, T), dtype=torch.bool, device=device), x.clone()

    h_rel = h_final.index_select(1, idx)            # (1, n_rel, d)
    h_norm = ln_f(h_rel)
    logits_sp = ff_out(h_norm)                      # (1, n_rel, V)
    if scale_logits:
        logits_sp = logits_sp.mul(1.0 / math.sqrt(d_model))

    p_sp = torch.softmax(logits_sp.to(torch.float32), dim=-1)
    x0_sp = logits_sp.argmax(dim=-1)                # (1, n_rel) long
    x0_p_sp = torch.gather(p_sp, dim=-1, index=x0_sp.unsqueeze(-1)).squeeze(-1)  # (1, n_rel)

    pass_thr = (x0_p_sp >= threshold)               # (1, n_rel) bool
    # Force the single max-confidence position to progress
    max_local = torch.argmax(x0_p_sp, dim=1, keepdim=True)  # (1, 1) — index into n_rel
    force = torch.zeros_like(pass_thr)
    force.scatter_(1, max_local, True)
    sparse_transfer = pass_thr | force              # (1, n_rel) bool

    # Scatter sparse → full shape
    transfer = torch.zeros((1, T), dtype=torch.bool, device=device)
    transfer.index_copy_(1, idx, sparse_transfer)

    x0 = x.clone()
    x0.index_copy_(1, idx, x0_sp)
    return transfer, x0


# ---- GSM8K answer extraction (from phase3_rankr_oracle) ----

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


# ---- Capture (block_in[0] + block_out per layer) ----

class AllLayerCapture:
    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.block_in: Dict[int, torch.Tensor] = {}
        self.block_out: Dict[int, torch.Tensor] = {}
        self._handles = []

    def install(self, model: Any):
        blocks = model.model.transformer.blocks
        for li, block in enumerate(blocks):
            def make_pre(layer_id):
                def hook(module, args):
                    self.block_in[layer_id] = args[0].detach()
                    return None
                return hook
            def make_post(layer_id):
                def hook(module, args, output):
                    out_tensor = output[0] if isinstance(output, tuple) else output
                    self.block_out[layer_id] = out_tensor.detach()
                    return None
                return hook
            self._handles.append(block.register_forward_pre_hook(make_pre(li)))
            self._handles.append(block.register_forward_hook(make_post(li)))

    def snapshot_block_in_0(self) -> torch.Tensor:
        return self.block_in[0].clone()

    def snapshot_block_out_all(self) -> List[torch.Tensor]:
        return [self.block_out[l].clone() for l in range(self.n_layers)]


# ---- Hybrid forward returning per-layer intermediates ----

def hybrid_cascade_forward_with_intermediates(
    blocks,
    n_layers: int,
    cur_block_in_0: torch.Tensor,                  # current step's embedding (layer 0 input)
    prev_block_out_per_layer: List[torch.Tensor],  # length N, each (1, T, D)
    A_per_layer: List[torch.Tensor],               # length N, each (k,) long indices
):
    """Run hybrid cascade forward, return:
       - final hidden (1, T, D)
       - list of N hybrid block_out states (the "consistent state" to cache as next-step prev).
    Refresh A_l positions, reuse prev_block_out[l] for the rest."""
    hybrid = cur_block_in_0.clone()
    new_block_outs: List[torch.Tensor] = []
    for l in range(n_layers):
        with torch.inference_mode():
            out = blocks[l](hybrid, attention_bias=None)
            out_l = out[0] if isinstance(out, tuple) else out
        A_l = A_per_layer[l]
        nxt = prev_block_out_per_layer[l].clone()
        if A_l.numel() > 0:
            nxt[0, A_l, :] = out_l[0, A_l, :]
        new_block_outs.append(nxt.clone())
        hybrid = nxt
    return hybrid, new_block_outs


def window_set_indices(U: torch.Tensor, w: int, T: int, device) -> torch.Tensor:
    if U.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    offsets = torch.arange(-w, w + 1, device=device).view(1, -1)
    grid = (U.view(-1, 1) + offsets).reshape(-1).clamp(0, T - 1)
    return torch.unique(grid)


def build_zar_A(
    block_start: int,
    block_end: int,
    prompt_len: int,
    T: int,
    lookahead: int,
    prompt_tail: int,
    device,
) -> torch.Tensor:
    """Zone-Aware Refresh set, single tensor (same A used for all layers).

    Refreshed regions:
      - current block:     [block_start, block_end)
      - lookahead:         [block_end, min(block_end + lookahead, T))
      - prompt tail:       [max(0, prompt_len - prompt_tail), prompt_len)

    All three concatenated, deduped, sorted ascending.
    """
    parts = []
    if prompt_tail > 0:
        parts.append(torch.arange(max(0, prompt_len - prompt_tail), prompt_len,
                                  dtype=torch.long, device=device))
    parts.append(torch.arange(block_start, block_end, dtype=torch.long, device=device))
    if lookahead > 0 and block_end < T:
        parts.append(torch.arange(block_end, min(block_end + lookahead, T),
                                  dtype=torch.long, device=device))
    return torch.unique(torch.cat(parts))


def build_E3_A_per_layer(
    lag1_sorted: List[torch.Tensor],
    U_t: torch.Tensor,
    K: int,
    T: int,
    device,
    w: int = 4,
    window_only: bool = False,
) -> List[torch.Tensor]:
    """A = U_t ∪ window(U_t, w) first, then fill with top-(K-...) by lag-1.

    If window_only=True, skip the lag-1 fill stage entirely and return only
    the window set (no K cap). Used for the v4 ablation that tests whether
    frontier-local refresh alone is sufficient.
    """
    W_idx = window_set_indices(U_t, w, T, device)
    if window_only:
        return [W_idx for _ in lag1_sorted]
    W_set = set(W_idx.cpu().tolist())
    A_per_layer: List[torch.Tensor] = []
    for idx in lag1_sorted:
        out = list(W_idx.cpu().tolist())[:K]
        if len(out) < K:
            for p in idx.cpu().tolist():
                if p in W_set:
                    continue
                out.append(p)
                if len(out) >= K:
                    break
        A_per_layer.append(torch.tensor(out[:K], dtype=torch.long, device=device))
    return A_per_layer


# ---- Inference loop ----

def run_sample(
    model,
    tokenizer,
    capture: AllLayerCapture,
    sample: Dict[str, Any],
    demo_ds,
    args,
    device,
):
    blocks = model.model.transformer.blocks
    n_layers = int(model.config.n_layers)
    ln_f = model.model.transformer.ln_f
    ff_out = model.model.transformer.ff_out
    scale_logits = bool(model.config.scale_logits)
    d_model = int(model.config.d_model)

    prompt_text = get_prompt_text(tokenizer, sample, demo_ds, args.num_fewshot, no_chat_template=False)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    prompt_len = int(input_ids.shape[1])
    total_len = prompt_len + args.gen_length

    x = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    num_blocks = args.gen_length // args.block_length
    steps_per_block = args.steps // num_blocks

    prev_snap: Optional[List[torch.Tensor]] = None
    prev_prev_snap: Optional[List[torch.Tensor]] = None
    prev_transfer: Optional[torch.Tensor] = None
    k_cache: Optional[List[torch.Tensor]] = None
    v_cache: Optional[List[torch.Tensor]] = None
    last_warmup_block_in_per_layer: Optional[List[torch.Tensor]] = None
    last_warmup_block_in_0: Optional[torch.Tensor] = None
    global_step = 0

    t0 = time.perf_counter()
    n_cheap_steps = 0
    n_baseline_steps = 0

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
            if block_mask.sum() == 0:
                break
            # In parallel-decoding mode, ignore k_step schedule; threshold drives unmask
            use_parallel = getattr(args, "threshold", 0.0) > 0.0
            if not use_parallel:
                k_step = int(num_transfer[0, i].item())
                if k_step <= 0:
                    break

            T_len = int(x.shape[1])
            use_cheap = (
                args.mode == "cheap_e3"
                and global_step >= args.warmup
                and prev_snap is not None
                and prev_prev_snap is not None
                and prev_transfer is not None
            )

            if not use_cheap:
                # Full baseline forward (warmup, or baseline mode)
                with torch.inference_mode():
                    out = model(x)
                logits_step = out.logits
                cur_block_in_0 = capture.snapshot_block_in_0()
                cur_block_out_all = capture.snapshot_block_out_all()
                n_baseline_steps += 1
                new_prev = cur_block_out_all
                # Track block_in per layer for KV-cache init on first cheap step.
                # block_in[l] = block_out[l-1] for l>0, or block_in[0] for l=0
                last_warmup_block_in_0 = cur_block_in_0
                last_warmup_block_in_per_layer = [cur_block_in_0] + cur_block_out_all[:-1]
            else:
                # Cheap hybrid forward — first need cur_embed (block_in[0])
                # The hooks fire on any forward; we ALSO need cur_block_in_0 to start hybrid.
                # Trick: run forward via model.transformer.wte to get the embedding cheaply.
                wte = model.model.transformer.wte
                with torch.inference_mode():
                    cur_block_in_0 = wte(x).detach()

                # Compute lag-1 sorted per layer (from prev - prev_prev)
                lag1_sorted: List[torch.Tensor] = []
                for l in range(n_layers):
                    d1 = (prev_snap[l].to(torch.float32)
                          - prev_prev_snap[l].to(torch.float32)).norm(dim=-1)[0]
                    lag1_sorted.append(torch.argsort(d1, descending=True))

                U_t = prev_transfer[0].nonzero(as_tuple=True)[0]
                # Per-layer K schedule (each layer may use a different K_l).
                # Default: uniform args.K. Otherwise parse from args.k_schedule.
                if getattr(args, "k_schedule", "uniform") == "uniform":
                    K_per_layer = [args.K] * n_layers
                elif args.k_schedule == "stepped":
                    # Scale relative to args.K (mid-layer base): shallow=0.5K, mid=K, very-deep=0.75K
                    K_per_layer = ([args.K // 2] * 8 + [args.K] * 16 + [int(args.K * 0.75)] * 8)
                elif args.k_schedule == "linear":
                    # Linear ramp from K_min to K_max with layer depth
                    K_min, K_max = 128, 512
                    K_per_layer = [int(K_min + (K_max - K_min) * l / max(n_layers - 1, 1))
                                   for l in range(n_layers)]
                elif args.k_schedule.startswith("custom:"):
                    # custom:K0,K1,...,K31 (comma-separated)
                    K_per_layer = [int(x) for x in args.k_schedule.split(":", 1)[1].split(",")]
                    assert len(K_per_layer) == n_layers, f"k_schedule must have {n_layers} entries"
                else:
                    raise ValueError(f"unknown k_schedule: {args.k_schedule}")

                A_per_layer = []
                for l, K_l in enumerate(K_per_layer):
                    A_l = build_E3_A_per_layer(
                        [lag1_sorted[l]], U_t, min(K_l, T_len), T_len, device, w=args.window
                    )[0]
                    A_per_layer.append(A_l)

                if getattr(args, "use_kv_cache", False):
                    # Lazy init KV cache from last warmup step's block_in
                    if k_cache is None:
                        k_cache, v_cache = init_kv_cache_from_block_inputs(
                            blocks, n_layers, last_warmup_block_in_per_layer,
                        )
                    h_final, new_prev = sparse_hybrid_cascade_forward_with_kv(
                        blocks, n_layers, cur_block_in_0, prev_snap, A_per_layer,
                        k_cache, v_cache, U_t,
                    )
                elif getattr(args, "use_sparse_block", False):
                    attn_fn = {
                        "fallback": masked_attention_fallback,
                        "sdpa": masked_attention_sdpa,
                        "sparse_q": masked_attention_sdpa_sparse_q,
                    }[getattr(args, "masked_attn", "sparse_q")]
                    h_final, new_prev = sparse_hybrid_cascade_forward(
                        blocks, n_layers, cur_block_in_0, prev_snap, A_per_layer,
                        masked_attention_fn=attn_fn,
                    )
                else:
                    h_final, new_prev = hybrid_cascade_forward_with_intermediates(
                        blocks, n_layers, cur_block_in_0, prev_snap, A_per_layer,
                    )
                # Compute logits from final hidden
                with torch.inference_mode():
                    h_norm = ln_f(h_final)
                    logits_step = ff_out(h_norm)
                    if scale_logits:
                        logits_step = logits_step.mul(1.0 / math.sqrt(d_model))
                n_cheap_steps += 1

            if use_parallel:
                transfer, x0 = pick_transfer_threshold(logits_step, x, block_mask, args.threshold)
            else:
                transfer, x0 = pick_transfer(logits_step, x, block_mask, k_step)
            x = torch.where(transfer, x0, x)
            prev_prev_snap = prev_snap
            prev_snap = new_prev
            prev_transfer = transfer.clone()
            global_step += 1
            if (x[:, block_start:block_end] == MASK_ID).sum() == 0:
                break

    wall = time.perf_counter() - t0

    # Decode generation
    gen_ids = x[0, prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    gold_raw = sample.get("answer", "")
    gold = normalize_num(extract_gold_answer(gold_raw))
    pred = normalize_num(extract_pred_answer(gen_text))
    correct = (gold is not None and pred is not None and gold == pred)

    return {
        "gen_text": gen_text,
        "gold": gold,
        "pred": pred,
        "correct": bool(correct),
        "wall_s": wall,
        "n_cheap_steps": n_cheap_steps,
        "n_baseline_steps": n_baseline_steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--start-sample", type=int, default=0)
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--mode", type=str, choices=["baseline", "cheap_e3"], required=True)
    p.add_argument("--K", type=int, default=384)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2,
                   help="number of full baseline steps before switching to cheap")
    p.add_argument("--use-sparse-block", action="store_true",
                   help="route cheap forward through sparse_block.py (kernel-attachable)")
    p.add_argument("--masked-attn", type=str, choices=["fallback", "sdpa", "sparse_q"], default="sparse_q",
                   help="L0 fallback | L1 SDPA (Q slice after RoPE) | L2 sparse_q (Q proj on A_l only)")
    p.add_argument("--use-kv-cache", action="store_true",
                   help="L3 — incremental K/V cache (only recompute K/V at A_{l-1, t} positions)")
    p.add_argument("--k-schedule", type=str, default="uniform",
                   help="uniform | stepped | linear | custom:K0,K1,...,K31")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="If >0, use Fast-dLLM parallel decoding (unmask positions with confidence >= threshold)")
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[phase14a] loading model ({args.mode}) ...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLaDAModelLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    n_layers = int(model.config.n_layers)
    capture = AllLayerCapture(n_layers)
    capture.install(model)

    test_ds = load_gsm8k(args.seed, args.split)
    demo_ds = load_gsm8k(args.seed, "train") if args.num_fewshot > 0 else None

    out_jsonl = os.path.join(args.out_dir, "results.jsonl")
    f_out = open(out_jsonl, "w", encoding="utf-8")

    sample_indices = list(range(args.start_sample, args.start_sample + args.num_samples))
    n_correct = 0
    total_wall = 0.0

    for idx in sample_indices:
        sample = test_ds[idx]
        res = run_sample(model, tokenizer, capture, sample, demo_ds, args, device)
        row = {
            "sample_id": int(idx),
            "mode": args.mode,
            "K": int(args.K),
            "window": int(args.window),
            "warmup": int(args.warmup),
            "gold": res["gold"],
            "pred": res["pred"],
            "correct": res["correct"],
            "wall_s": res["wall_s"],
            "n_cheap_steps": res["n_cheap_steps"],
            "n_baseline_steps": res["n_baseline_steps"],
            "gen_text": res["gen_text"][:1500],  # truncate for jsonl
        }
        f_out.write(json.dumps(row) + "\n")
        f_out.flush()
        if res["correct"]:
            n_correct += 1
        total_wall += res["wall_s"]
        status = "✓" if res["correct"] else "✗"
        print(f"[phase14a] sample {idx:>4} {status}  pred={res['pred']} gold={res['gold']}  "
              f"{res['wall_s']:.1f}s  cheap/base={res['n_cheap_steps']}/{res['n_baseline_steps']}",
              flush=True)

    f_out.close()
    n = len(sample_indices)
    acc = n_correct / max(n, 1)
    print(f"[phase14a] DONE: {n_correct}/{n} = {acc:.3f} accuracy  total {total_wall:.0f}s",
          flush=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"n": n, "n_correct": n_correct, "accuracy": acc,
                   "total_wall_s": total_wall}, f, indent=2)


if __name__ == "__main__":
    main()
