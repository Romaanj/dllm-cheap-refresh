"""
Dream-7B lm-evaluation-harness adapter.

Supports two modes:
  - baseline:  full forward 256 times with greedy/parallel unmask
  - cheap_e3:  Phase 14a cheap forward (sparse Q + KV cache + lag-1 estimator)

Both modes optionally combine with parallel decoding via `threshold`.

Usage:
  accelerate launch --num_processes 1 eval_dream.py \\
    --tasks gsm8k --num_fewshot 5 --batch_size 1 \\
    --confirm_run_unsafe_code --model dream_dist \\
    --model_args "model_path=Dream-org/Dream-v0-Instruct-7B,gen_length=256,steps=256,block_length=32,cheap_e3=True,cheap_K=256,cheap_k_schedule=uniform,cheap_warmup=2,threshold=0.9,show_speed=True" \\
    --output_path results_lmeval_dream_cheap/dream_composition_full1319 \\
    --log_samples
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import accelerate
import torch
import torch.nn.functional as F
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/inverse_cdf/Fast-dLLM/dream")
from model.modeling_dream import DreamModel  # type: ignore
from model.configuration_dream import DreamConfig  # type: ignore

from phase14a_dream import (
    DREAM_MASK_ID,
    DreamAllLayerCapture,
    sparse_dream_hybrid_cascade,
    init_dream_kv_cache,
    build_E3_A_gpu,
    build_W_and_mask,
    get_num_transfer_tokens,
    pick_transfer_greedy,
    pick_transfer_threshold,
)


def dream_generate_baseline(
    model, input_ids,
    *, gen_length=256, block_length=32, steps=256,
    threshold=0.0, mask_id=DREAM_MASK_ID,
):
    """Full Dream forward at every step (no cheap). Returns (x, nfe)."""
    assert input_ids.shape[0] == 1
    device = input_ids.device
    prompt_len = int(input_ids.shape[1])
    total_len = prompt_len + gen_length
    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    use_parallel = threshold > 0.0
    nfe = 0
    for nb in range(num_blocks):
        block_start = prompt_len + nb * block_length
        block_end = prompt_len + (nb + 1) * block_length
        block_mask_init = (x[:, block_start:block_end] == mask_id)
        num_transfer = get_num_transfer_tokens(block_mask_init, steps_per_block)
        for i in range(steps_per_block):
            cur_mask = (x == mask_id)
            block_mask = cur_mask.clone()
            block_mask[:, :block_start] = False
            block_mask[:, block_end:] = False
            if block_mask.sum() == 0:
                break
            if not use_parallel:
                k_step = int(num_transfer[0, i].item())
                if k_step <= 0:
                    break
            with torch.inference_mode():
                out = model(x)
            logits_step = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
            nfe += 1
            if use_parallel:
                transfer, x0 = pick_transfer_threshold(logits_step, x, block_mask, threshold)
            else:
                transfer, x0 = pick_transfer_greedy(logits_step, x, block_mask, k_step, mask_id)
            x = torch.where(transfer, x0, x)
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
    return x, nfe


def dream_generate_cheap_e3(
    model, capture, input_ids,
    *, K=256, k_schedule="uniform", warmup=2, threshold=0.0, window=4,
    gen_length=256, block_length=32, steps=256, mask_id=DREAM_MASK_ID,
):
    """Dream cheap E3 inference. Returns (x, nfe)."""
    assert input_ids.shape[0] == 1
    device = input_ids.device
    prompt_len = int(input_ids.shape[1])
    total_len = prompt_len + gen_length
    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    layers = model.model.layers
    n_layers = len(layers)
    norm_f = model.model.norm
    lm_head = model.lm_head
    embed = model.model.embed_tokens

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    use_parallel = threshold > 0.0
    prev_snap = None
    prev_prev_snap = None
    prev_transfer = None
    k_cache = None
    v_cache = None
    last_warmup_block_in_per_layer = None
    global_step = 0
    n_cheap = 0
    n_base = 0
    for nb in range(num_blocks):
        block_start = prompt_len + nb * block_length
        block_end = prompt_len + (nb + 1) * block_length
        block_mask_init = (x[:, block_start:block_end] == mask_id)
        num_transfer = get_num_transfer_tokens(block_mask_init, steps_per_block)
        for i in range(steps_per_block):
            cur_mask = (x == mask_id)
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
            do_cheap = (
                global_step >= warmup
                and prev_snap is not None and prev_prev_snap is not None and prev_transfer is not None
            )
            if not do_cheap:
                with torch.inference_mode():
                    out = model(x)
                logits_step = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
                new_prev = capture.snapshot_block_out_all()
                cur_block_in_0 = capture.snapshot_block_in_0()
                last_warmup_block_in_per_layer = [cur_block_in_0] + new_prev[:-1]
                n_base += 1
            else:
                with torch.inference_mode():
                    cur_block_in_0 = embed(x).detach()
                # Batched lag-1 ranking
                prev_stack = torch.stack(prev_snap, dim=0)
                prev_prev_stack = torch.stack(prev_prev_snap, dim=0)
                d1_all = (prev_stack.to(torch.float32) - prev_prev_stack.to(torch.float32)).norm(dim=-1)[:, 0, :]
                lag1_sorted = list(torch.argsort(d1_all, dim=-1, descending=True))
                U_t = prev_transfer[0].nonzero(as_tuple=True)[0]
                if k_schedule == "uniform":
                    K_per_layer = [K] * n_layers
                elif k_schedule == "stepped":
                    third = n_layers // 3
                    K_per_layer = ([K // 2] * third + [K] * third + [int(K * 0.75)] * (n_layers - 2 * third))
                else:
                    raise ValueError(k_schedule)
                W_step, W_mask_step = build_W_and_mask(U_t, window, T_len, device)
                A_per_layer = [build_E3_A_gpu(lag1_sorted[l], W_step, W_mask_step, min(K_l, T_len), T_len)
                               for l, K_l in enumerate(K_per_layer)]
                if k_cache is None:
                    k_cache, v_cache = init_dream_kv_cache(layers, n_layers, last_warmup_block_in_per_layer)
                h_final, new_prev = sparse_dream_hybrid_cascade(
                    layers, n_layers, cur_block_in_0, prev_snap, A_per_layer, k_cache, v_cache, U_t,
                )
                with torch.inference_mode():
                    h_norm = norm_f(h_final)
                    logits_full = lm_head(h_norm)
                    logits_step = torch.cat([logits_full[:, :1], logits_full[:, :-1]], dim=1)
                n_cheap += 1
            if use_parallel:
                transfer, x0 = pick_transfer_threshold(logits_step, x, block_mask, threshold)
            else:
                transfer, x0 = pick_transfer_greedy(logits_step, x, block_mask, k_step, mask_id)
            x = torch.where(transfer, x0, x)
            prev_prev_snap = prev_snap
            prev_snap = new_prev
            prev_transfer = transfer.clone()
            global_step += 1
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
    return x, n_base + n_cheap


@register_model("dream_dist")
class DreamEvalHarness(LM):
    def __init__(
        self,
        model_path="Dream-org/Dream-v0-Instruct-7B",
        mask_id=DREAM_MASK_ID,
        max_length=4096,
        batch_size=1,
        steps=256,
        gen_length=256,
        block_length=32,
        device="cuda",
        threshold=0.0,
        save_dir=None,
        show_speed=False,
        cheap_e3=False,
        cheap_K=256,
        cheap_k_schedule="uniform",
        cheap_warmup=2,
        cheap_window=4,
        **kwargs,
    ):
        super().__init__()
        acc = accelerate.Accelerator()
        self.accelerator = acc if acc.num_processes > 1 else None
        config = DreamConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = DreamModel.from_pretrained(
            model_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mask_id = mask_id
        self.batch_size = int(batch_size)
        self.max_length = max_length
        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.threshold = float(threshold)
        self.save_dir = save_dir
        self.show_speed = bool(show_speed) if not isinstance(show_speed, str) else show_speed.lower() in ("true", "1", "yes")
        self.is_instruct = "instruct" in model_path.lower()

        self.cheap_e3 = bool(cheap_e3) if not isinstance(cheap_e3, str) else cheap_e3.lower() in ("true", "1", "yes")
        self.cheap_K = int(cheap_K)
        self.cheap_k_schedule = str(cheap_k_schedule)
        self.cheap_warmup = int(cheap_warmup)
        self.cheap_window = int(cheap_window)
        self._cheap_capture = None
        if self.cheap_e3:
            n_layers_ = int(config.num_hidden_layers)
            self._cheap_capture = DreamAllLayerCapture(n_layers_)
            self._cheap_capture.install(self.model)
            print(f"[eval_dream] cheap_e3 ENABLED: K={self.cheap_K} schedule={self.cheap_k_schedule} "
                  f"warmup={self.cheap_warmup} window={self.cheap_window} threshold={self.threshold}",
                  flush=True)
        else:
            print(f"[eval_dream] baseline mode (no cheap), threshold={self.threshold}", flush=True)

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests):
        raise NotImplementedError("Dream adapter currently supports only generate_until")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"rank_{self.rank}.jsonl")
            if os.path.exists(save_path):
                import json
                with open(save_path) as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)

        start_time = time.time()
        for i, req in enumerate(tqdm(requests, desc="Generating...")):
            if i < processed_count:
                continue
            question = req.args[0]
            # Apply chat template uniformly for instruct models (incl. HumanEval),
            # matching Fast-dLLM upstream + Elastic-Cache convention.
            if self.is_instruct:
                m = [{"role": "user", "content": question}]
                user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            else:
                user_input = question
            input_ids = torch.tensor(self.tokenizer(user_input, add_special_tokens=True)["input_ids"],
                                     dtype=torch.long, device=self.device).unsqueeze(0)

            stop_tokens = req.args[1].get("until", [])

            if self.cheap_e3:
                generated_answer, nfe = dream_generate_cheap_e3(
                    self.model, self._cheap_capture, input_ids,
                    K=self.cheap_K, k_schedule=self.cheap_k_schedule,
                    warmup=self.cheap_warmup, threshold=self.threshold,
                    window=self.cheap_window,
                    gen_length=self.gen_length, block_length=self.block_length, steps=self.steps,
                    mask_id=self.mask_id,
                )
            else:
                generated_answer, nfe = dream_generate_baseline(
                    self.model, input_ids,
                    gen_length=self.gen_length, block_length=self.block_length, steps=self.steps,
                    threshold=self.threshold, mask_id=self.mask_id,
                )

            gen_ids = generated_answer[0, input_ids.shape[1]:]
            if self.show_speed:
                num_tokens += (gen_ids != self.mask_id).sum().item()
                num_nfe += nfe
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            for stop_seq in stop_tokens:
                if stop_seq in gen_text:
                    gen_text = gen_text.split(stop_seq)[0]
            gen_ids2 = torch.tensor(self.tokenizer(gen_text)["input_ids"])
            gen_text = self.tokenizer.decode(gen_ids2, skip_special_tokens=True)
            output.append(gen_text)
            if self.save_dir is not None:
                import json
                with open(save_path, "a") as f:
                    f.write(json.dumps(gen_text, ensure_ascii=False) + "\n")
            print(f"answer: {gen_text}\nnfe: {nfe}\navg_nfe: {num_nfe/max(len(output),1):.1f}\n", flush=True)

        end_time = time.time()
        if self.show_speed:
            print(f"Total tokens: {num_tokens}")
            print(f"Total time: {end_time - start_time:.1f}s")
            print(f"Tokens/sec: {num_tokens/(end_time - start_time):.2f}")
            print(f"Total NFE: {num_nfe}")
        return output


if __name__ == "__main__":
    cli_evaluate()
