"""Measure how K/V projections at different position classes evolve during
baseline (full-forward, no caching) LLaDA decoding.

Hypothesis (user): after the first 32-64 unmask steps, prompt-side K/V
stabilizes because the generation direction is set early.

We capture k_proj and v_proj outputs at every layer at every decode step,
then compute drift metrics:
  - drift_from_warmup(t, l, class) = 1 - cos(K_t[l, p], K_0[l, p])  averaged over p in class
  - drift_lag1(t, l, class)        = 1 - cos(K_t[l, p], K_{t-1}[l, p])

Position classes:
  - prompt:          p in [0, Lp)
  - gen_unmasked:    p in [Lp, T), x[p] != MASK at current step
  - gen_masked:      p in [Lp, T), x[p] == MASK at current step

Output: jsonl of per-(sample, step, layer, class) records to a results dir.
A separate plotting script can aggregate and visualize.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

MASK_ID = 126336


# -----------------------------
# K/V projection capture hooks
# -----------------------------

class KVCapture:
    """Captures the output of every block's k_proj and v_proj."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.k: List[Optional[torch.Tensor]] = [None] * n_layers
        self.v: List[Optional[torch.Tensor]] = [None] * n_layers
        self.handles = []

    def attach(self, model):
        blocks = model.model.transformer.blocks
        for l, block in enumerate(blocks):
            def mk_hook(idx, target):
                def hook(_module, _inp, out):
                    target[idx] = out.detach()
                return hook
            self.handles.append(block.k_proj.register_forward_hook(mk_hook(l, self.k)))
            self.handles.append(block.v_proj.register_forward_hook(mk_hook(l, self.v)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# -----------------------------
# Prompt builders
# -----------------------------

def build_he_prompt(tokenizer, problem_prompt: str) -> str:
    """HumanEval 0-shot using LLaDA chat template."""
    msg = [{"role": "user", "content": problem_prompt}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def build_gsm8k_prompt(tokenizer, fewshot_examples, question: str) -> str:
    """GSM8K 5-shot in lm-eval format with chat template."""
    text = ""
    for ex in fewshot_examples:
        text += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    text += f"Question: {question}\nAnswer:"
    msg = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


# -----------------------------
# Drift loop
# -----------------------------

@torch.inference_mode()
def run_one(
    model,
    tokenizer,
    capture: KVCapture,
    prompt_text: str,
    sample_id: str,
    task: str,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    threshold: float = 0.9,
    device: str = "cuda",
):
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    Lp = int(input_ids.shape[1])
    T = Lp + gen_length

    x = torch.full((1, T), MASK_ID, dtype=torch.long, device=device)
    x[:, :Lp] = input_ids

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    K0: Optional[List[torch.Tensor]] = None
    V0: Optional[List[torch.Tensor]] = None
    Kprev: Optional[List[torch.Tensor]] = None
    Vprev: Optional[List[torch.Tensor]] = None

    records: List[Dict] = []
    step_idx = 0

    for nb in range(num_blocks):
        block_start = Lp + nb * block_length
        block_end = block_start + block_length

        for i in range(steps_per_block):
            block_mask = (x == MASK_ID).clone()
            block_mask[:, :block_start] = False
            block_mask[:, block_end:] = False
            if block_mask.sum() == 0:
                break

            # Full baseline forward
            out = model(x)
            logits = out.logits

            # Snapshot K/V (still in bf16 from model)
            K_curr = [k.clone() for k in capture.k]
            V_curr = [v.clone() for v in capture.v]

            if K0 is None:
                K0 = [k.clone() for k in K_curr]
                V0 = [v.clone() for v in V_curr]

            # Position classes at current step (BEFORE applying transfer)
            is_mask = (x == MASK_ID)[0]
            arange = torch.arange(T, device=device)
            prompt_pos = arange[:Lp]
            gen_mask = arange >= Lp
            gen_unmasked_pos = arange[gen_mask & ~is_mask]
            gen_masked_pos   = arange[gen_mask & is_mask]

            # Prompt sub-bins by distance from gen boundary (Lp - 1).
            # dist = Lp - 1 - p for p in [0, Lp). So prompt[-32:] has dist [0, 32).
            # Edges (open at the top): [0,32), [32,128), [128,512), [512, +inf)
            dist_from_gen = (Lp - 1 - prompt_pos)
            prompt_d0_32      = prompt_pos[(dist_from_gen >= 0) & (dist_from_gen < 32)]
            prompt_d32_128    = prompt_pos[(dist_from_gen >= 32) & (dist_from_gen < 128)]
            prompt_d128_512   = prompt_pos[(dist_from_gen >= 128) & (dist_from_gen < 512)]
            prompt_d512_plus  = prompt_pos[dist_from_gen >= 512]

            # Compute drift per (layer, class)
            for l in range(len(K_curr)):
                K_t = K_curr[l][0]  # (T, d_kv)
                V_t = V_curr[l][0]
                K_w = K0[l][0]
                V_w = V0[l][0]
                K_p = Kprev[l][0] if Kprev is not None else None
                V_p = Vprev[l][0] if Vprev is not None else None

                for class_name, pos in (("prompt", prompt_pos),
                                        ("prompt_d0_32",     prompt_d0_32),
                                        ("prompt_d32_128",   prompt_d32_128),
                                        ("prompt_d128_512",  prompt_d128_512),
                                        ("prompt_d512_plus", prompt_d512_plus),
                                        ("gen_unmasked", gen_unmasked_pos),
                                        ("gen_masked", gen_masked_pos)):
                    n = int(pos.numel())
                    if n == 0:
                        continue
                    # Cast to fp32 for stable cosine
                    Kt = K_t.index_select(0, pos).float()
                    Vt = V_t.index_select(0, pos).float()
                    Kw = K_w.index_select(0, pos).float()
                    Vw = V_w.index_select(0, pos).float()

                    cos_K_w = F.cosine_similarity(Kt, Kw, dim=-1).mean().item()
                    cos_V_w = F.cosine_similarity(Vt, Vw, dim=-1).mean().item()

                    cos_K_lag = float("nan")
                    cos_V_lag = float("nan")
                    if K_p is not None:
                        Kpv = K_p.index_select(0, pos).float()
                        Vpv = V_p.index_select(0, pos).float()
                        cos_K_lag = F.cosine_similarity(Kt, Kpv, dim=-1).mean().item()
                        cos_V_lag = F.cosine_similarity(Vt, Vpv, dim=-1).mean().item()

                    # Magnitudes (also useful — drift might be in scale)
                    K_norm = Kt.norm(dim=-1).mean().item()
                    V_norm = Vt.norm(dim=-1).mean().item()

                    records.append({
                        "task": task,
                        "sample_id": sample_id,
                        "step": step_idx,
                        "block": nb,
                        "layer": l,
                        "class": class_name,
                        "n_pos": n,
                        "drift_K_from_warmup": 1.0 - cos_K_w,
                        "drift_V_from_warmup": 1.0 - cos_V_w,
                        "drift_K_lag1":        1.0 - cos_K_lag,
                        "drift_V_lag1":        1.0 - cos_V_lag,
                        "K_norm_mean": K_norm,
                        "V_norm_mean": V_norm,
                    })

            # Parallel decoding: unmask positions with confidence >= threshold,
            # PLUS force the single max-confidence position to progress
            # (matches phase14a_cheap_e2e.pick_transfer_threshold).
            probs = F.softmax(logits.float(), dim=-1)
            top_p, top_idx = probs.max(dim=-1)
            neg_inf = torch.full_like(top_p, float("-inf"))
            conf = torch.where(block_mask, top_p, neg_inf)
            transfer = block_mask & (top_p >= threshold)
            max_idx = torch.argmax(conf, dim=1, keepdim=True)
            force = torch.zeros_like(transfer).scatter_(1, max_idx, True)
            transfer = (transfer | force) & block_mask
            x = torch.where(transfer, top_idx, x)

            Kprev, Vprev = K_curr, V_curr
            step_idx += 1

            if (x[:, block_start:block_end] == MASK_ID).sum() == 0:
                break

    return records, Lp


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output jsonl path")
    p.add_argument("--task", choices=["he", "gsm8k"], required=True)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--gen_length", type=int, default=128)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    args = p.parse_args()

    device = "cuda"
    print(f"Loading model {args.model_path} …")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to(device).eval()
    n_layers = int(model.config.n_layers)
    capture = KVCapture(n_layers)
    capture.attach(model)

    # Build prompts
    prompts = []
    if args.task == "he":
        ds = load_dataset("openai_humaneval", split="test")
        for i in range(args.n_samples):
            ex = ds[i]
            ptext = build_he_prompt(tok, ex["prompt"])
            prompts.append((f"he_{ex['task_id']}", ptext))
    else:
        ds_train = load_dataset("gsm8k", "main", split="train")
        ds_test = load_dataset("gsm8k", "main", split="test")
        rng = torch.Generator().manual_seed(1234)
        idx_train = torch.randperm(len(ds_train), generator=rng)[:5].tolist()
        few = [ds_train[i] for i in idx_train]
        for i in range(args.n_samples):
            ex = ds_test[i]
            ptext = build_gsm8k_prompt(tok, few, ex["question"])
            prompts.append((f"gsm8k_{i}", ptext))

    # Run
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.perf_counter()
    with open(args.out, "w") as f:
        for sid, ptext in prompts:
            t1 = time.perf_counter()
            recs, Lp = run_one(
                model, tok, capture, ptext, sid, args.task,
                gen_length=args.gen_length, steps=args.steps,
                block_length=args.block_length, threshold=args.threshold,
                device=device,
            )
            elapsed = time.perf_counter() - t1
            print(f"  sample {sid}: Lp={Lp}, {len(recs)} records, {elapsed:.1f}s")
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print(f"Total {time.perf_counter() - t0:.1f}s. Saved {args.out}")
    capture.detach()


if __name__ == "__main__":
    main()
