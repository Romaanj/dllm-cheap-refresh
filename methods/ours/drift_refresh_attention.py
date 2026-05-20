"""V0 prototype: drift-based refresh cache.

Per (layer, head, query):
  - Cache previous step's attention vector and attention output.
  - Each new step, compute attention as normal.
  - For each (h, q): if cos sim with cached attn > 1 - threshold → reuse cached output.
                    else → use new output, update cache.

V0 has a SINGLE threshold for all heads. Cluster-aware thresholds = V1.

Goal: validate that drift-based gating outperforms static scope (V1 from
scope_incremental_attention). Drift signal should be naturally cluster-aware
(stationary_sink has low drift → reuse often; mask_binder has high drift → refresh often).
"""

import math
import types
import string

import torch
import torch.nn.functional as F
import numpy as np


def apply_rotary_at_positions(rotary_emb, x, positions):
    seq_len = rotary_emb.config.max_sequence_length
    pos_sin, pos_cos = rotary_emb.get_rotary_embedding(seq_len, x.device)
    pos_sin = pos_sin.type_as(x)
    pos_cos = pos_cos.type_as(x)
    pos_sin_sel = pos_sin.index_select(2, positions)
    pos_cos_sel = pos_cos.index_select(2, positions)
    return rotary_emb.apply_rotary_pos_emb(pos_sin_sel, pos_cos_sel, x)


class DriftRefreshCache:
    """Single-threshold drift-based refresh cache."""

    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.attn_cache = {}    # layer_idx → (H, T, L) prev attention
        self.out_cache = {}     # layer_idx → (B, H, T, hd) prev output
        self.last_T = None
        self.stats = {"reused": 0, "recomputed": 0}

    def reset(self):
        self.attn_cache = {}
        self.out_cache = {}
        self.last_T = None

    def is_new_generation(self, current_T):
        return self.last_T is None or self.last_T != current_T

    def gate(self, layer_idx, attn_curr, output_curr):
        """attn_curr: (H, T, L) FP16/FP32 attention.
        output_curr: (B, H, T, hd) attention output.
        Returns gated output: cached where drift < threshold, new otherwise.
        """
        prev_attn = self.attn_cache.get(layer_idx)
        prev_out = self.out_cache.get(layer_idx)
        if prev_attn is None or prev_out is None or prev_attn.shape != attn_curr.shape:
            self.attn_cache[layer_idx] = attn_curr.detach().to(torch.float16).clone()
            self.out_cache[layer_idx] = output_curr.detach().clone()
            return output_curr

        # Cos sim per (h, q): (H, T, L) vs (H, T, L) → (H, T)
        a1 = prev_attn.to(attn_curr.dtype)
        a2 = attn_curr
        a1n = a1 / a1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        a2n = a2 / a2.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        cos_sim = (a1n * a2n).sum(dim=-1)  # (H, T)
        drift = 1.0 - cos_sim
        reuse_mask = (drift < self.threshold)  # (H, T) bool

        # Gate output: (B, H, T, hd). reuse_mask (H, T) → broadcast (1, H, T, 1)
        m = reuse_mask.unsqueeze(0).unsqueeze(-1)
        new_out = torch.where(m, prev_out, output_curr)

        # Update cache (always store latest)
        self.attn_cache[layer_idx] = attn_curr.detach().to(torch.float16).clone()
        self.out_cache[layer_idx] = new_out.detach().clone()

        # Stats
        self.stats["reused"] += int(reuse_mask.sum().item())
        self.stats["recomputed"] += int((~reuse_mask).sum().item())
        return new_out

    def commit(self, current_T):
        self.last_T = current_T


def make_patched_attention(layer_idx):
    def _attn(self, q, k, v, mask=None, attention_bias=None,
              layer_past=None, use_cache=False,
              replace_position=None, output_attentions=False):
        if attention_bias is None and mask is not None:
            attention_bias = mask
            mask = None

        cache = self._drift_cache
        B, T, C = q.size()
        n_heads = self.config.n_heads
        head_dim = C // n_heads
        scale = 1.0 / math.sqrt(head_dim)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=k.dtype)
            k = self.k_norm(k).to(dtype=k.dtype)

        Q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        K = k.view(B, T, n_heads, head_dim).transpose(1, 2)
        V = v.view(B, T, n_heads, head_dim).transpose(1, 2)

        positions = torch.arange(T, device=q.device, dtype=torch.long)
        Q = apply_rotary_at_positions(self.rotary_emb, Q, positions)
        K = apply_rotary_at_positions(self.rotary_emb, K, positions)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if attention_bias is not None:
            scores = scores + attention_bias
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn, V)  # (B, H, T, hd)

        # Drift gate
        attn_for_drift = attn.squeeze(0)  # (H, T, L)
        gated = cache.gate(layer_idx, attn_for_drift, output)

        gated = gated.transpose(1, 2).contiguous().view(B, T, C)
        gated = self.attn_out(gated)
        return gated, None, attn if output_attentions else None

    return _attn


def install_drift_refresh(model, threshold=0.1):
    cache = DriftRefreshCache(threshold=threshold)
    blocks = list(model.model.transformer.blocks)
    for li, block in enumerate(blocks):
        block._drift_cache = cache
        block._layer_idx = li
        block.attention = types.MethodType(make_patched_attention(li), block)

    orig_forward = model.forward

    def patched_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if input_ids is not None:
            T = input_ids.shape[-1]
            if cache.is_new_generation(T):
                cache.reset()
            ret = orig_forward(*args, **kwargs)
            cache.commit(T)
            return ret
        return orig_forward(*args, **kwargs)

    model.forward = patched_forward
    print(f"[drift-refresh] threshold={threshold} installed on {len(blocks)} layers")
    return cache


def reset_cache(model):
    blocks = list(model.model.transformer.blocks)
    if blocks and hasattr(blocks[0], "_drift_cache"):
        blocks[0]._drift_cache.reset()
