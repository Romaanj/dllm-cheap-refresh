"""
LLaDA cheap E3 inference, packaged for use as a drop-in for `generate()`.

Returns (generated_answer, nfe) — same signature as LLaDA's standard `generate`.
Designed to be called from eval_llada.py's generate_until.

The function expects batch_size=1 and `input_ids` of shape (1, prompt_len).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F

from phase14a_cheap_e2e import (
    AllLayerCapture,
    hybrid_cascade_forward_with_intermediates,
    build_E3_A_per_layer,
    pick_transfer_threshold,
)
from phase9_day3_composed import get_num_transfer_tokens, pick_transfer, MASK_ID
from sparse_block import (
    sparse_hybrid_cascade_forward,
    sparse_hybrid_cascade_forward_with_kv,
    init_kv_cache_from_block_inputs,
    masked_attention_sdpa_sparse_q,
)


def generate_cheap_e3(
    model,
    capture: AllLayerCapture,
    input_ids: torch.Tensor,
    *,
    K: int = 384,
    k_schedule: str = "stepped",
    warmup: int = 2,
    threshold: float = 0.0,
    window: int = 4,
    gen_length: int = 256,
    block_length: int = 32,
    steps: int = 256,
    use_kv_cache: bool = True,
    use_sparse_block: bool = True,
    mask_id: int = MASK_ID,
):
    """LLaDA cheap E3 + (optional) parallel decoding.

    Returns:
        generated_answer: (1, prompt_len + gen_length) long tensor
        nfe: number of FORWARDS (full baseline + cheap)
    """
    assert input_ids.shape[0] == 1, "cheap_e3 currently batch_size=1 only"
    device = input_ids.device
    blocks = model.model.transformer.blocks
    n_layers = int(model.config.n_layers)
    ln_f = model.model.transformer.ln_f
    ff_out = model.model.transformer.ff_out
    scale_logits = bool(model.config.scale_logits)
    d_model = int(model.config.d_model)

    prompt_len = int(input_ids.shape[1])
    total_len = prompt_len + gen_length
    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    use_parallel = threshold > 0.0

    prev_snap: Optional[List[torch.Tensor]] = None
    prev_prev_snap: Optional[List[torch.Tensor]] = None
    prev_transfer: Optional[torch.Tensor] = None
    k_cache: Optional[List[torch.Tensor]] = None
    v_cache: Optional[List[torch.Tensor]] = None
    last_warmup_block_in_per_layer: Optional[List[torch.Tensor]] = None
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
            use_cheap = (
                global_step >= warmup
                and prev_snap is not None
                and prev_prev_snap is not None
                and prev_transfer is not None
            )

            if not use_cheap:
                with torch.inference_mode():
                    out = model(x)
                logits_step = out.logits
                new_prev = capture.snapshot_block_out_all()
                last_warmup_block_in_per_layer = [capture.snapshot_block_in_0()] + new_prev[:-1]
                n_base += 1
            else:
                wte = model.model.transformer.wte
                with torch.inference_mode():
                    cur_block_in_0 = wte(x).detach()
                lag1_sorted = []
                for l in range(n_layers):
                    d1 = (prev_snap[l].to(torch.float32)
                          - prev_prev_snap[l].to(torch.float32)).norm(dim=-1)[0]
                    lag1_sorted.append(torch.argsort(d1, descending=True))
                U_t = prev_transfer[0].nonzero(as_tuple=True)[0]
                if k_schedule == "uniform":
                    K_per_layer = [K] * n_layers
                elif k_schedule == "stepped":
                    K_per_layer = ([K // 2] * 8 + [K] * 16 + [int(K * 0.75)] * 8)
                else:
                    raise ValueError(f"unknown k_schedule: {k_schedule}")
                A_per_layer = []
                for l, K_l in enumerate(K_per_layer):
                    A_l = build_E3_A_per_layer(
                        [lag1_sorted[l]], U_t, min(K_l, T_len), T_len, device, w=window
                    )[0]
                    A_per_layer.append(A_l)

                if use_kv_cache:
                    if k_cache is None:
                        k_cache, v_cache = init_kv_cache_from_block_inputs(
                            blocks, n_layers, last_warmup_block_in_per_layer,
                        )
                    h_final, new_prev = sparse_hybrid_cascade_forward_with_kv(
                        blocks, n_layers, cur_block_in_0, prev_snap, A_per_layer,
                        k_cache, v_cache, U_t,
                    )
                elif use_sparse_block:
                    h_final, new_prev = sparse_hybrid_cascade_forward(
                        blocks, n_layers, cur_block_in_0, prev_snap, A_per_layer,
                        masked_attention_fn=masked_attention_sdpa_sparse_q,
                    )
                else:
                    h_final, new_prev = hybrid_cascade_forward_with_intermediates(
                        blocks, n_layers, cur_block_in_0, prev_snap, A_per_layer,
                    )
                with torch.inference_mode():
                    h_norm = ln_f(h_final)
                    logits_step = ff_out(h_norm)
                    if scale_logits:
                        logits_step = logits_step.mul(1.0 / math.sqrt(d_model))
                n_cheap += 1

            if use_parallel:
                transfer, x0 = pick_transfer_threshold(logits_step, x, block_mask, threshold)
            else:
                transfer, x0 = pick_transfer(logits_step, x, block_mask, k_step)
            x = torch.where(transfer, x0, x)
            prev_prev_snap = prev_snap
            prev_snap = new_prev
            prev_transfer = transfer.clone()
            global_step += 1
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

    return x, n_base + n_cheap
