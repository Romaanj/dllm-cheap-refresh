"""
Phase-12 oracle — Dream-7B port.

Same probe as phase12_cascade_oracle.py but on Dream-v0-Instruct-7B:
for each step pair (t-1, t), capture block_in / block_out at ALL 28 layers,
then for each K in the grid run a hybrid cascade forward (refresh top-K
per layer by ||cur_block_out - prev_block_out||, reuse prev_block_out for
the rest) and measure cos_logits / argmax_agreement vs baseline.

Cross-model diagnostic: does Dream show the same cascade containment as
LLaDA (cos≥0.998 at K=128, argmax≥0.96 at K=384)? If not, the cheap mode
accuracy drop on Dream has an explanation rooted in Dream's geometry, not
the lag-1 estimator.

Usage:
  CUDA_VISIBLE_DEVICES=0 python phase12_cascade_oracle_dream.py \\
    --num-samples 4 --probe-stride 16 \\
    --k-values 0,32,64,128,256,384,512,768,1024,-1 \\
    --out-dir results_phase12_cascade_oracle_dream/gpu0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/inverse_cdf/Fast-dLLM/dream")
from model.modeling_dream import DreamModel  # type: ignore
from model.configuration_dream import DreamConfig  # type: ignore

DREAM_MASK_ID = 151666


# -------------------- Inlined GSM8K helpers (avoid LLaDA model conflict) --------------------

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


def pick_transfer_greedy(logits, x, block_mask, k):
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


# -------------------- Dream layer capture --------------------

class DreamAllLayerCapture:
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.block_in: Dict[int, torch.Tensor] = {}
        self.block_out: Dict[int, torch.Tensor] = {}
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

    def snapshot_all(self) -> Dict[str, List[torch.Tensor]]:
        return {
            "block_in": [self.block_in[l].clone() for l in range(self.n_layers)],
            "block_out": [self.block_out[l].clone() for l in range(self.n_layers)],
        }

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# -------------------- Hybrid cascade forward (Dream) --------------------

def dream_hybrid_cascade_forward(
    layers,
    n_layers: int,
    cur_block_in_0: torch.Tensor,
    prev_block_out_per_layer: List[torch.Tensor],
    A_per_layer: List[torch.Tensor],
    position_embeddings,
    position_ids,
    return_mixed_per_layer: bool = False,
):
    """Hybrid forward: full attention at each layer, then refresh A_l / reuse prev_block_out elsewhere.

    If return_mixed_per_layer, also returns list of per-layer MIXED outputs
    (== prev_block_out scatter-A_l fresh out_l). Useful for cheap-mode lag-1 tracking.
    """
    hybrid = cur_block_in_0.clone()
    mixed_outs = []
    for l in range(n_layers):
        with torch.inference_mode():
            out = layers[l](
                hybrid,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            out_l = out[0] if isinstance(out, tuple) else out
        A_l = A_per_layer[l]
        nxt = prev_block_out_per_layer[l].clone()
        if A_l.numel() > 0:
            nxt[0, A_l, :] = out_l[0, A_l, :]
        if return_mixed_per_layer:
            mixed_outs.append(nxt.clone())
        hybrid = nxt
    if return_mixed_per_layer:
        return hybrid, mixed_outs
    return hybrid


# -------------------- Per-layer stats --------------------

def compute_per_layer_stats(cur_block_out, prev_block_out, n_layers, T_len):
    layer_total_norm = []
    layer_norm_top16 = []
    layer_norm_top64 = []
    layer_norm_top256 = []
    layer_active_count_relmean = {0.5: [], 1.0: [], 2.0: [], 4.0: []}
    for l in range(n_layers):
        delta = (cur_block_out[l].to(torch.float32) - prev_block_out[l].to(torch.float32))
        per_pos_norm = delta.norm(dim=-1)[0]
        total = float(per_pos_norm.sum().item())
        layer_total_norm.append(total)
        sorted_idx = torch.argsort(per_pos_norm, descending=True)
        for K, store in zip([16, 64, 256], [layer_norm_top16, layer_norm_top64, layer_norm_top256]):
            top_sum = float(per_pos_norm[sorted_idx[:min(K, T_len)]].sum().item())
            store.append(top_sum / max(total, 1e-30))
        mean_norm = per_pos_norm.mean().item()
        for ratio in layer_active_count_relmean.keys():
            cnt = int((per_pos_norm > ratio * mean_norm).sum().item())
            layer_active_count_relmean[ratio].append(cnt)
    return {
        "layer_total_norm": layer_total_norm,
        "layer_norm_coverage_top16": layer_norm_top16,
        "layer_norm_coverage_top64": layer_norm_top64,
        "layer_norm_coverage_top256": layer_norm_top256,
        "layer_active_count_relmean_0p5": layer_active_count_relmean[0.5],
        "layer_active_count_relmean_1p0": layer_active_count_relmean[1.0],
        "layer_active_count_relmean_2p0": layer_active_count_relmean[2.0],
        "layer_active_count_relmean_4p0": layer_active_count_relmean[4.0],
    }


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
    p.add_argument("--probe-stride", type=int, default=16)
    p.add_argument("--k-values", default="0,32,64,128,256,384,512,768,1024,-1")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-probes-per-sample", type=int, default=10**9)
    args = p.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[phase12-dream] loading {args.model}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DreamConfig.from_pretrained(args.model, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.model, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    n_layers = int(config.num_hidden_layers)
    layers = model.model.layers
    norm_f = model.model.norm
    lm_head = model.lm_head
    rotary_emb = model.model.rotary_emb
    print(f"[phase12-dream] loaded; n_layers={n_layers}", flush=True)

    capture = DreamAllLayerCapture(n_layers)
    capture.install(model)

    test_ds = load_gsm8k(args.seed, args.split)
    demo_ds = load_gsm8k(args.seed, "train") if args.num_fewshot > 0 else None

    out_jsonl = os.path.join(args.out_dir, "cascade_oracle.jsonl")
    f_out = open(out_jsonl, "w", encoding="utf-8")

    sample_indices = list(range(args.start_sample, args.start_sample + args.num_samples))

    for sample_id in sample_indices:
        sample = test_ds[sample_id]
        prompt_text = get_prompt_text(tokenizer, sample, demo_ds, args.num_fewshot, no_chat_template=False)
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc.input_ids.to(device)
        prompt_len = int(input_ids.shape[1])
        total_len = prompt_len + args.gen_length

        x = torch.full((1, total_len), DREAM_MASK_ID, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids

        num_blocks = args.gen_length // args.block_length
        steps_per_block = args.steps // num_blocks

        prev_snap: Optional[Dict[str, List[torch.Tensor]]] = None
        global_step = 0
        n_probes_done = 0
        t_start = time.perf_counter()

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
                k_step = int(num_transfer[0, i].item())
                if k_step <= 0 or block_mask.sum() == 0:
                    break

                with torch.inference_mode():
                    out = model(x)
                logits_baseline = out.logits
                cur_snap = capture.snapshot_all()
                T_len = int(x.shape[1])

                block_mask_full = block_mask[0]
                n_block_masks = int(block_mask_full.sum().item())

                should_probe = (
                    prev_snap is not None
                    and global_step > 0
                    and n_block_masks > 0
                    and (global_step % args.probe_stride == 0)
                    and n_probes_done < args.max_probes_per_sample
                )

                if should_probe:
                    # Two views: (a) RAW logits at block_mask (model output, unshifted)
                    # (b) SHIFTED logits at block_mask = next-token convention applied,
                    # which is what greedy decoding actually consumes on Dream.
                    logits_baseline_shift = torch.cat(
                        [logits_baseline[:, :1], logits_baseline[:, :-1]], dim=1
                    )
                    logits_baseline_masked = logits_baseline[0, block_mask_full, :].to(torch.float32)
                    logits_baseline_shift_masked = logits_baseline_shift[0, block_mask_full, :].to(torch.float32)
                    argmax_baseline = logits_baseline_masked.argmax(dim=-1)
                    argmax_baseline_shift = logits_baseline_shift_masked.argmax(dim=-1)
                    top5_baseline = logits_baseline_masked.topk(5, dim=-1).indices
                    top5_baseline_shift = logits_baseline_shift_masked.topk(5, dim=-1).indices

                    cur_block_in_0 = cur_snap["block_in"][0]
                    cur_block_out_all = cur_snap["block_out"]
                    prev_block_out_all = prev_snap["block_out"]

                    # Pre-compute position embeddings once for this T_len
                    position_ids = torch.arange(T_len, device=device, dtype=torch.long).unsqueeze(0)
                    with torch.inference_mode():
                        position_embeddings = rotary_emb(cur_block_in_0, position_ids)

                    # Per-layer sorted_indices (oracle ranking)
                    layer_sorted_idx: List[torch.Tensor] = []
                    layer_per_pos_norm: List[torch.Tensor] = []
                    for l in range(n_layers):
                        delta = (cur_block_out_all[l].to(torch.float32)
                                 - prev_block_out_all[l].to(torch.float32))
                        per_pos_norm = delta.norm(dim=-1)[0]
                        layer_per_pos_norm.append(per_pos_norm)
                        layer_sorted_idx.append(torch.argsort(per_pos_norm, descending=True))

                    layer_stats = compute_per_layer_stats(
                        cur_block_out_all, prev_block_out_all, n_layers, T_len
                    )

                    for k_raw in k_values:
                        k_eff = T_len if k_raw < 0 else max(0, int(k_raw))
                        k_eff = min(k_eff, T_len)

                        A_per_layer = [layer_sorted_idx[l][:k_eff] for l in range(n_layers)]

                        h_final = dream_hybrid_cascade_forward(
                            layers, n_layers, cur_block_in_0,
                            prev_block_out_all, A_per_layer,
                            position_embeddings, position_ids,
                        )

                        with torch.inference_mode():
                            h_norm = norm_f(h_final)
                            logits_hybrid_full = lm_head(h_norm)
                        logits_hybrid_shift_full = torch.cat(
                            [logits_hybrid_full[:, :1], logits_hybrid_full[:, :-1]], dim=1
                        )
                        logits_hybrid = logits_hybrid_full[0, block_mask_full, :].to(torch.float32)
                        logits_hybrid_shift = logits_hybrid_shift_full[0, block_mask_full, :].to(torch.float32)
                        cos_logit = F.cosine_similarity(
                            logits_hybrid.reshape(-1).unsqueeze(0),
                            logits_baseline_masked.reshape(-1).unsqueeze(0),
                        ).item()
                        cos_logit_shift = F.cosine_similarity(
                            logits_hybrid_shift.reshape(-1).unsqueeze(0),
                            logits_baseline_shift_masked.reshape(-1).unsqueeze(0),
                        ).item()
                        argmax_hyb = logits_hybrid.argmax(dim=-1)
                        argmax_hyb_shift = logits_hybrid_shift.argmax(dim=-1)
                        argmax_match = (argmax_hyb == argmax_baseline).float().mean().item()
                        argmax_match_shift = (argmax_hyb_shift == argmax_baseline_shift).float().mean().item()
                        top5_match = (argmax_hyb.unsqueeze(-1) == top5_baseline).any(dim=-1).float().mean().item()
                        top5_match_shift = (argmax_hyb_shift.unsqueeze(-1) == top5_baseline_shift).any(dim=-1).float().mean().item()

                        baseline_final = cur_block_out_all[n_layers - 1]
                        ha = h_final.to(torch.float32)[0].reshape(-1)
                        hb = baseline_final.to(torch.float32)[0].reshape(-1)
                        cos_hidden = F.cosine_similarity(ha.unsqueeze(0), hb.unsqueeze(0)).item()

                        coverage_per_layer = []
                        for l in range(n_layers):
                            tot = float(layer_per_pos_norm[l].sum().item())
                            top_sum = (float(layer_per_pos_norm[l][layer_sorted_idx[l][:k_eff]].sum().item())
                                       if k_eff > 0 else 0.0)
                            coverage_per_layer.append(top_sum / max(tot, 1e-30))

                        row = {
                            "sample_id": int(sample_id),
                            "global_step": int(global_step),
                            "block_id": int(nb),
                            "block_step": int(i),
                            "T": int(T_len),
                            "n_block_masks": int(n_block_masks),
                            "K_uniform": int(k_eff),
                            "K_frac": float(k_eff) / float(T_len),
                            "cos_logits_block_masks": float(cos_logit),
                            "argmax_agreement": float(argmax_match),
                            "top5_agreement": float(top5_match),
                            "cos_logits_shift_block_masks": float(cos_logit_shift),
                            "argmax_agreement_shift": float(argmax_match_shift),
                            "top5_agreement_shift": float(top5_match_shift),
                            "cos_hidden_final": float(cos_hidden),
                            "delta_norm_coverage_per_layer": coverage_per_layer,
                            "layer_stats": layer_stats,
                        }
                        f_out.write(json.dumps(row) + "\n")
                    f_out.flush()
                    n_probes_done += 1

                # Dream: greedy decoding consumes shifted logits (next-token convention)
                logits_for_decode = torch.cat(
                    [logits_baseline[:, :1], logits_baseline[:, :-1]], dim=1
                )
                transfer, x0 = pick_transfer_greedy(logits_for_decode, x, block_mask, k_step)
                x = torch.where(transfer, x0, x)
                prev_snap = cur_snap
                global_step += 1
                if (x[:, block_start:block_end] == DREAM_MASK_ID).sum() == 0:
                    break

        wall = time.perf_counter() - t_start
        print(f"[phase12-dream] sample {sample_id} done: {n_probes_done} probes, {wall:.1f}s", flush=True)

    f_out.close()
    print("[phase12-dream] DONE", flush=True)


if __name__ == "__main__":
    main()
