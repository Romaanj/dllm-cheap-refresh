"""Sparse-aware LLaDA block forward — refactor scaffold for kernel attachment.

Provides `sparse_llada_block_forward` that replaces the standard block forward
in our cheap E3 hybrid cascade, exposing 4 clear "kernel slots" where we can
later drop in custom Triton kernels.

Correctness invariant:
  When A_l = arange(L), sparse_llada_block_forward(block, x, A_l, prev) must
  equal block(x) regardless of prev. This is the sanity test.

Compute reduction:
  - FFN: O(|A_l| × d × ffn_hidden) instead of O(L × d × ffn_hidden)
  - Attention output projection: O(|A_l| × d²) instead of O(L × d²)
  - Residual + scatter: O(|A_l| × d)

Future kernel slots:
  SLOT 1: sparse Q projection + RoPE (q_proj only on A_l rows, RoPE with explicit pos)
  SLOT 2: incremental K, V (only recompute for A_prev positions, reuse cache)
  SLOT 3: masked attention (Q from A_l only, full K/V) — BIGGEST gain (~K/T × full attn)
  SLOT 4: fused SwiGLU FFN (single kernel: ff_norm + ff_proj + up_proj + act + mul + ff_out)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# =====================================================================
# KERNEL SLOT 3 — Masked attention
# Levels:
#   L0 (fallback):        full attention then slice A_l. No saving.
#   L1 (SDPA):            full Q proj + RoPE, slice Q[A_l], SDPA. Saves attn compute.
#   L2 (SDPA + sparse Q): Q proj only on A_l rows + RoPE at A_l positions, SDPA.
#                         Adds Q proj saving on top of L1.
#   L3 (Triton):          fused tile kernel (future).
#
# All variants take (block, x_normed_full, k_full, v_full, A_l). The function
# internally chooses how to compute Q. This API allows future sparse K/V too.
# =====================================================================

def _apply_rope_at_positions(block, q_heads, positions, L):
    """Apply LLaDA RoPE to q_heads (B, H, |A|, dh) at given positions (|A|,)."""
    cfg = block.config
    if not cfg.rope:
        return q_heads
    # Use the block's rotary_emb cached sin/cos
    pos_sin, pos_cos = block.rotary_emb.get_rotary_embedding(L, q_heads.device)
    if cfg.rope_full_precision:
        q_work = q_heads.float()
    else:
        q_work = q_heads
    with torch.autocast(q_heads.device.type, enabled=False):
        pos_sin = pos_sin.type_as(q_work)
        pos_cos = pos_cos.type_as(q_work)
        pos_sin_A = pos_sin.index_select(2, positions)             # (1, 1, |A|, dh)
        pos_cos_A = pos_cos.index_select(2, positions)
        q_rope = block.rotary_emb.apply_rotary_pos_emb(pos_sin_A, pos_cos_A, q_work)
    return q_rope.type_as(q_heads)


def masked_attention_fallback(
    block,
    x_normed: torch.Tensor,      # (1, L, d) — attn_norm output, full
    k_full: torch.Tensor,        # (1, L, d_kv) — output of k_proj
    v_full: torch.Tensor,        # (1, L, d_kv) — output of v_proj
    A_l: torch.Tensor,
) -> torch.Tensor:
    """L0 fallback: compute full attention then slice A_l rows.
    Wasteful in compute but correctness-preserving."""
    q_full = block.q_proj(x_normed)
    att, _, _ = block.attention(q_full, k_full, v_full, attention_bias=None)
    return att[:, A_l, :]


def masked_attention_sdpa(
    block,
    x_normed: torch.Tensor,      # (1, L, d) — full
    k_full: torch.Tensor,        # (1, L, d_kv)
    v_full: torch.Tensor,        # (1, L, d_kv)
    A_l: torch.Tensor,
) -> torch.Tensor:
    """L1 — Full Q proj + RoPE, slice Q[A_l], SDPA.
    Saves attention compute (|A|/L) but Q projection still full."""
    cfg = block.config
    n_heads = cfg.n_heads
    n_kv = cfg.effective_n_kv_heads
    B, L, C = x_normed.shape
    head_dim = C // n_heads

    q_full = block.q_proj(x_normed)                                # (B, L, d)  ← full proj
    q = q_full.view(B, L, n_heads, head_dim).transpose(1, 2)       # (B, H, L, dh)
    k = k_full.view(B, L, n_kv, head_dim).transpose(1, 2)
    v = v_full.view(B, L, n_kv, head_dim).transpose(1, 2)

    if cfg.rope:
        q, k = block.rotary_emb(q, k)                              # RoPE on full Q, K

    if n_heads != n_kv:
        k_exp = k.repeat_interleave(n_heads // n_kv, dim=1)
        v_exp = v.repeat_interleave(n_heads // n_kv, dim=1)
    else:
        k_exp, v_exp = k, v

    q_A = q.index_select(2, A_l)                                   # (B, H, |A|, dh)
    att = F.scaled_dot_product_attention(q_A, k_exp, v_exp, attn_mask=None, dropout_p=0.0, is_causal=False)
    att = att.transpose(1, 2).contiguous().view(B, A_l.shape[0], C)
    return block.attn_out(att)


# ----- L2/L3 helper: sparse Q with pre-projected K/V (KV-cache path) -----

def masked_attention_sdpa_sparse_q_cached_kv(
    block,
    x_normed: torch.Tensor,      # (1, L, d) — full
    k_full: torch.Tensor,        # (1, L, d_kv) — cached/updated K (pre-RoPE)
    v_full: torch.Tensor,        # (1, L, d_kv) — cached V (RoPE not applied to V)
    A_l: torch.Tensor,
) -> torch.Tensor:
    """Same as masked_attention_sdpa_sparse_q but takes pre-projected K, V
    (e.g. from KV cache). Saves K/V proj compute."""
    cfg = block.config
    n_heads = cfg.n_heads
    n_kv = cfg.effective_n_kv_heads
    B, L, C = x_normed.shape
    head_dim = C // n_heads
    n_A = int(A_l.shape[0])

    # Sparse Q proj at A_l rows
    x_A = x_normed.index_select(1, A_l)
    q_A_raw = block.q_proj(x_A)
    q_A_heads = q_A_raw.view(B, n_A, n_heads, head_dim).transpose(1, 2)  # (B, H, |A|, dh)

    # K, V already projected (pre-RoPE)
    k = k_full.view(B, L, n_kv, head_dim).transpose(1, 2)
    v = v_full.view(B, L, n_kv, head_dim).transpose(1, 2)

    # RoPE: Q at A_l positions, K at all L positions
    q_A_heads = _apply_rope_at_positions(block, q_A_heads, A_l, L)
    if cfg.rope:
        all_pos = torch.arange(L, device=k.device, dtype=torch.long)
        k = _apply_rope_at_positions(block, k, all_pos, L)

    if n_heads != n_kv:
        k_exp = k.repeat_interleave(n_heads // n_kv, dim=1)
        v_exp = v.repeat_interleave(n_heads // n_kv, dim=1)
    else:
        k_exp, v_exp = k, v

    att = F.scaled_dot_product_attention(q_A_heads, k_exp, v_exp, attn_mask=None, dropout_p=0.0, is_causal=False)
    att = att.transpose(1, 2).contiguous().view(B, n_A, C)
    return block.attn_out(att)


def masked_attention_sdpa_sparse_q(
    block,
    x_normed: torch.Tensor,      # (1, L, d) — full
    k_full: torch.Tensor,        # (1, L, d_kv)
    v_full: torch.Tensor,        # (1, L, d_kv)
    A_l: torch.Tensor,
) -> torch.Tensor:
    """L2 — Sparse Q proj on A_l rows + RoPE at A_l positions.
    Saves on Q projection (|A|/L) in addition to L1's attention saving.

    Q proj cost: O(|A| × d²)  vs L1's O(L × d²) — saves ~K/T × 21 GFLOPS/layer.
    K/V projection still full (incremental K/V update is a future kernel slot).
    """
    cfg = block.config
    n_heads = cfg.n_heads
    n_kv = cfg.effective_n_kv_heads
    B, L, C = x_normed.shape
    head_dim = C // n_heads
    n_A = int(A_l.shape[0])

    # Sparse Q proj: only |A| rows × d² GEMM
    x_A = x_normed.index_select(1, A_l)                            # (B, |A|, d)
    q_A_raw = block.q_proj(x_A)                                    # (B, |A|, d)
    q_A_heads = q_A_raw.view(B, n_A, n_heads, head_dim).transpose(1, 2)  # (B, H, |A|, dh)

    # K, V on full L (still needed as attention keys/values)
    k = k_full.view(B, L, n_kv, head_dim).transpose(1, 2)
    v = v_full.view(B, L, n_kv, head_dim).transpose(1, 2)

    # Apply RoPE: Q at A_l positions, K at all L positions
    q_A_heads = _apply_rope_at_positions(block, q_A_heads, A_l, L)
    if cfg.rope:
        # For K full, reuse block.rotary_emb but only apply to K (need to call on dummy q)
        # Simpler: apply RoPE to K directly using the same helper with positions = arange(L)
        all_pos = torch.arange(L, device=k.device, dtype=torch.long)
        # apply rotary to k via the helper (works for any tensor shaped (B, H, T, dh))
        k = _apply_rope_at_positions(block, k, all_pos, L)

    if n_heads != n_kv:
        k_exp = k.repeat_interleave(n_heads // n_kv, dim=1)
        v_exp = v.repeat_interleave(n_heads // n_kv, dim=1)
    else:
        k_exp, v_exp = k, v

    att = F.scaled_dot_product_attention(q_A_heads, k_exp, v_exp, attn_mask=None, dropout_p=0.0, is_causal=False)
    att = att.transpose(1, 2).contiguous().view(B, n_A, C)
    return block.attn_out(att)


# =====================================================================
# KERNEL SLOT 4 — Fused sparse SwiGLU FFN
# Default: PyTorch ops on A_l rows. Already gives ~K/T × FFN speedup.
# =====================================================================

def sparse_ffn_fallback(
    block,
    y_attn_A: torch.Tensor,      # (1, |A_l|, d) — pre-FFN input at A_l rows
) -> torch.Tensor:
    """PyTorch-native sparse FFN. Operates only on A_l rows."""
    h = block.ff_norm(y_attn_A)
    x_ffn = block.ff_proj(h)
    x_up = block.up_proj(h)
    x_ffn = block.act(x_ffn) * x_up
    x_ffn = block.ff_out(x_ffn)
    return x_ffn                  # (1, |A_l|, d)


# =====================================================================
# Main sparse block forward
# =====================================================================

def sparse_llada_block_forward(
    block,                                          # LLaDALlamaBlock instance
    hybrid_input: torch.Tensor,                     # (1, L, d) — fresh at A_prev, stale elsewhere
    A_l: torch.Tensor,                              # (k,) long — refresh-output indices at THIS layer
    prev_block_out: torch.Tensor,                   # (1, L, d) — cached layer-l output from prev step
    masked_attention_fn=masked_attention_fallback,  # KERNEL SLOT 3 (overridable)
    sparse_ffn_fn=sparse_ffn_fallback,              # KERNEL SLOT 4 (overridable)
) -> torch.Tensor:
    """Returns: new_block_out (1, L, d) = prev for ~A_l, fresh for A_l.

    Compute footprint (with default fallbacks):
      - attn_norm:    O(L × d)               (full; required for K/V)
      - Q/K/V proj:   O(L × d²)              (full; kernel slots 1,2 can sparsify)
      - attention:    O(L² × d_h)            (full via fallback; slot 3 is the win)
      - att slicing:  O(|A_l| × d)
      - residual A_l: O(|A_l| × d)
      - sparse FFN:   O(|A_l| × d × ffn_h)   (sparse — saves K/T × FFN)
      - scatter:      O(|A_l| × d)
    """
    # ----- attention path -----
    x_normed = block.attn_norm(hybrid_input)                # (1, L, d)

    # KERNEL SLOT 2 (future): incremental K/V update only at A_prev positions
    k = block.k_proj(x_normed)                              # (1, L, d_kv)
    v = block.v_proj(x_normed)                              # (1, L, d_kv)

    # KERNEL SLOT 3: masked attention.  Q proj happens INSIDE attn_fn
    # (sparse variants project only on A_l rows).
    att_A = masked_attention_fn(block, x_normed, k, v, A_l) # (1, |A_l|, d)

    # Residual at A_l only
    y_attn_A = hybrid_input[:, A_l, :] + block.dropout(att_A)  # (1, |A_l|, d)

    # ----- FFN path (sparse) -----
    # KERNEL SLOT 4: fused sparse SwiGLU FFN
    ffn_out_A = sparse_ffn_fn(block, y_attn_A)              # (1, |A_l|, d)
    y_out_A = y_attn_A + block.dropout(ffn_out_A)           # (1, |A_l|, d)

    # ----- scatter into prev cache -----
    new_block_out = prev_block_out.clone()
    new_block_out[:, A_l, :] = y_out_A
    return new_block_out


# =====================================================================
# Sanity check: with A_l = arange(L), sparse path == reference forward
# =====================================================================

def sparse_block_correctness_test(
    block,
    x: torch.Tensor,         # (1, L, d)
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> dict:
    """When A_l covers all positions, sparse forward must equal reference block(x)."""
    with torch.inference_mode():
        out_ref = block(x, attention_bias=None)
        out_ref = out_ref[0] if isinstance(out_ref, tuple) else out_ref
        L = int(x.shape[1])
        A_l = torch.arange(L, dtype=torch.long, device=x.device)
        # prev_block_out is irrelevant here because A_l covers all
        prev_dummy = torch.zeros_like(x)
        out_sparse = sparse_llada_block_forward(block, x, A_l, prev_dummy)
    diff = (out_ref.float() - out_sparse.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cos = torch.nn.functional.cosine_similarity(
        out_ref.float().reshape(-1).unsqueeze(0),
        out_sparse.float().reshape(-1).unsqueeze(0),
    ).item()
    passed = max_abs < atol or cos > 1 - rtol
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "cos_similarity": cos,
        "passed": passed,
    }


# =====================================================================
# Convenience: cascade forward using sparse_block (drop-in for phase14a)
# =====================================================================

def sparse_hybrid_cascade_forward(
    blocks,
    n_layers: int,
    cur_block_in_0: torch.Tensor,                 # (1, L, d) — current embedding
    prev_block_out_per_layer: list,               # length N, each (1, L, d)
    A_per_layer: list,                            # length N, each (k,) long
    masked_attention_fn=masked_attention_fallback,
    sparse_ffn_fn=sparse_ffn_fallback,
) -> Tuple[torch.Tensor, list]:
    """Drop-in replacement for phase14a's hybrid_cascade_forward_with_intermediates.

    Returns (final_hidden, list of per-layer new block_outs).
    """
    hybrid = cur_block_in_0.clone()
    new_block_outs = []
    for l in range(n_layers):
        with torch.inference_mode():
            new_out = sparse_llada_block_forward(
                blocks[l], hybrid, A_per_layer[l], prev_block_out_per_layer[l],
                masked_attention_fn=masked_attention_fn,
                sparse_ffn_fn=sparse_ffn_fn,
            )
        new_block_outs.append(new_out.clone())
        hybrid = new_out
    return hybrid, new_block_outs


# =====================================================================
# KERNEL SLOT 2 — Incremental K, V cache
# Maintain per-layer K, V cache. Update only at A_input_changed positions
# (= A_{l-1, t} for layer l>0, = U_t for layer 0).
# Correctness: at non-A_input_changed positions, hybrid_input is bit-identical
# to previous step (we just copied the cached prev_block_out tensor), so K, V
# at those positions are unchanged. Saves K, V projection compute.
# =====================================================================

def sparse_llada_block_forward_with_kv_cache(
    block,
    hybrid_input: torch.Tensor,                     # (1, L, d)
    A_l: torch.Tensor,                              # (k,) — refresh set for this layer's OUTPUT
    A_input_changed: torch.Tensor,                  # (k_in,) — positions where input changed since prev step
    prev_block_out: torch.Tensor,                   # (1, L, d)
    k_cache: torch.Tensor,                          # (1, L, d_kv) — IN-PLACE updated
    v_cache: torch.Tensor,                          # (1, L, d_kv) — IN-PLACE updated
    masked_attention_fn=masked_attention_sdpa_sparse_q_cached_kv,
    sparse_ffn_fn=sparse_ffn_fallback,
) -> torch.Tensor:
    """Sparse block forward with K/V cache. Updates k_cache, v_cache IN PLACE."""
    x_normed = block.attn_norm(hybrid_input)

    # SLOT 2: incremental K/V update — only at A_input_changed positions
    if A_input_changed is not None and A_input_changed.numel() > 0:
        x_partial = x_normed.index_select(1, A_input_changed)            # (1, |Δ|, d)
        k_partial = block.k_proj(x_partial)                              # (1, |Δ|, d_kv)
        v_partial = block.v_proj(x_partial)
        k_cache.index_copy_(1, A_input_changed, k_partial)
        v_cache.index_copy_(1, A_input_changed, v_partial)
    # else: no input change, K/V cache is valid as-is

    # SLOT 3: masked attention using cached K, V
    att_A = masked_attention_fn(block, x_normed, k_cache, v_cache, A_l)

    # Residual + sparse FFN
    y_attn_A = hybrid_input[:, A_l, :] + block.dropout(att_A)
    ffn_out_A = sparse_ffn_fn(block, y_attn_A)
    y_out_A = y_attn_A + block.dropout(ffn_out_A)

    # Scatter
    new_block_out = prev_block_out.clone()
    new_block_out[:, A_l, :] = y_out_A
    return new_block_out


def init_kv_cache_from_block_inputs(
    blocks,
    n_layers: int,
    block_in_per_layer: list,                      # length N, each (1, L, d)
) -> Tuple[list, list]:
    """Compute initial K, V cache for all layers from given block_in tensors.
    Call once at the start of cheap inference (using warmup-step block_in)."""
    k_cache = []
    v_cache = []
    for l in range(n_layers):
        with torch.inference_mode():
            x_normed = blocks[l].attn_norm(block_in_per_layer[l])
            k_cache.append(blocks[l].k_proj(x_normed).clone())
            v_cache.append(blocks[l].v_proj(x_normed).clone())
    return k_cache, v_cache


def sparse_hybrid_cascade_forward_with_kv(
    blocks,
    n_layers: int,
    cur_block_in_0: torch.Tensor,                 # (1, L, d) — current embedding
    prev_block_out_per_layer: list,               # length N, each (1, L, d)
    A_per_layer: list,                            # length N, each (k,) long
    k_cache: list,                                # length N, each (1, L, d_kv) — IN-PLACE updated
    v_cache: list,                                # length N, each (1, L, d_kv) — IN-PLACE updated
    U_t: torch.Tensor,                            # (n_unmasked,) — A_input_changed for layer 0
    masked_attention_fn=masked_attention_sdpa_sparse_q_cached_kv,
    sparse_ffn_fn=sparse_ffn_fallback,
) -> Tuple[torch.Tensor, list]:
    """Sparse cascade forward with KV-cache reuse. Updates K/V cache in place."""
    hybrid = cur_block_in_0.clone()
    A_input_changed = U_t
    new_block_outs = []
    for l in range(n_layers):
        with torch.inference_mode():
            new_out = sparse_llada_block_forward_with_kv_cache(
                blocks[l], hybrid, A_per_layer[l], A_input_changed,
                prev_block_out_per_layer[l],
                k_cache[l], v_cache[l],
                masked_attention_fn=masked_attention_fn,
                sparse_ffn_fn=sparse_ffn_fn,
            )
        new_block_outs.append(new_out.clone())
        hybrid = new_out
        # For next layer, A_input_changed = this layer's refresh set
        A_input_changed = A_per_layer[l]
    return hybrid, new_block_outs
