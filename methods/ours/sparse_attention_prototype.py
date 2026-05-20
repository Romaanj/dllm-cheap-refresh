"""Sparse attention prototype: per-head cluster-specific attention mask.

Maps (layer, head) → cluster (from probe) → attention mask design:
  mask_binder:    keep mask positions ∪ window ±W_mb around active query
  punct_sink:     keep punct positions ∪ small window ±W_ps
  bidir_asym_L:   keep punct positions ∪ wide window ±W_bd
  diffuse:        full attention (fallback)

For correctness validation, this monkey-patches LLaDA attention to APPLY
the mask (not to skip compute). Future work: custom kernel for real speedup.
"""

import types
import math
import string

import torch
import torch.nn.functional as F
import numpy as np


CLUSTER_NAMES = [
    "stationary_sink", "frontier_tracker", "mask_binder", "mask_dropper",
    "broadening", "bidir_asym_L", "bidir_asym_R", "punct_sink", "other",
]
MB = CLUSTER_NAMES.index("mask_binder")
PS = CLUSTER_NAMES.index("punct_sink")
BD = CLUSTER_NAMES.index("bidir_asym_L")


# Default windows per cluster (more permissive — w/ R_curr ~ 55% keep target)
# Verification: p90 cost ~400 keys → use wider W for safety vs aggressive compression.
DEFAULT_W = {
    MB: 32,    # mask_binder: cover full ±block
    PS: 16,    # punct_sink: small structural window
    BD: 300,   # bidir_asym_L: wide left-recency
    # others / diffuse → full attention
}


def build_punct_special_id_sets(tokenizer):
    """Build punct + special token id sets (same logic as R_curr method)."""
    punct_chars = set(string.punctuation + "\n\t ")
    punct_ids = set()
    for tid in range(tokenizer.vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        if len(s) and all(c in punct_chars for c in s):
            punct_ids.add(tid)
    special_ids = set(tokenizer.all_special_ids or [])
    for s in ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
              "<|startoftext|>", "<|endoftext|>"]:
        x = tokenizer.encode(s, add_special_tokens=False)
        if len(x) == 1:
            special_ids.add(x[0])
    return punct_ids, special_ids


def make_sparse_attention(layer_idx, head_cluster_map, head_W_map,
                          punct_id_set, special_id_set, mask_id):
    """Build a sparse attention method for a given block (layer)."""

    def _sparse_attention(self, q, k, v, mask=None, attention_bias=None,
                          layer_past=None, use_cache=False,
                          replace_position=None, output_attentions=False):
        """Vanilla attention but with per-head sparse mask applied."""
        if attention_bias is None and mask is not None:
            attention_bias = mask
            mask = None

        B, T, C = q.size()
        dtype = k.dtype
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        n_heads = self.config.n_heads
        head_dim = C // n_heads
        scale = 1.0 / math.sqrt(head_dim)

        # Reshape to (B, H, T, hd)
        q_h = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        k_h = k.view(B, T, n_heads, head_dim).transpose(1, 2)
        v_h = v.view(B, T, n_heads, head_dim).transpose(1, 2)

        # Apply rotary
        from experiments.fastgen_head_probe.end2end_ragged_dual import apply_rotary_at_positions
        positions = torch.arange(T, device=q.device, dtype=torch.long)
        q_h = apply_rotary_at_positions(self.rotary_emb, q_h, positions)
        k_h = apply_rotary_at_positions(self.rotary_emb, k_h, positions)

        # Standard attention scores (B, H, T, T)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale

        # Build per-head sparse mask: (H, T, T) — True = ALLOWED to attend
        device = q.device
        token_ids = self._current_input_ids[0]  # (T,) — set by hook
        is_punct = torch.zeros(T, dtype=torch.bool, device=device)
        is_special = torch.zeros(T, dtype=torch.bool, device=device)
        is_mask = (token_ids == mask_id)
        if punct_id_set:
            punct_t = torch.as_tensor(list(punct_id_set), device=device)
            is_punct = torch.isin(token_ids, punct_t)
        if special_id_set:
            spec_t = torch.as_tensor(list(special_id_set), device=device)
            is_special = torch.isin(token_ids, spec_t)
        sink_punct = is_punct | is_special  # (T,) for punct-sink + bidir clusters
        sink_mask_pos = is_mask              # (T,) for mask_binder cluster
        is_mask_query = is_mask              # (T,) — TRUE where query is currently mask

        # Window: (T, T) — |i - j| <= W
        position_diff = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (T, T)

        # Per-head mask construction.
        # CRITICAL: apply sparse mask ONLY for mask queries (rows where query is mask token).
        # Content queries (already-decoded positions) keep FULL attention — they need it
        # to build proper prefix representations.
        head_masks = []
        all_true = torch.ones(T, T, dtype=torch.bool, device=device)
        for hi in range(n_heads):
            cluster = head_cluster_map.get((layer_idx, hi), -1)
            W = head_W_map.get((layer_idx, hi), None)
            if cluster == -1 or W is None:
                head_mask = all_true
            elif cluster == MB:
                in_window = position_diff <= W
                sparse_for_mask_q = in_window | sink_mask_pos.unsqueeze(0)
                head_mask = torch.where(
                    is_mask_query.unsqueeze(1),  # (T, 1) broadcast
                    sparse_for_mask_q,           # apply sparse only at mask query rows
                    all_true,                    # content query rows: full attention
                )
            elif cluster == PS:
                in_window = position_diff <= W
                sparse_for_mask_q = in_window | sink_punct.unsqueeze(0)
                head_mask = torch.where(is_mask_query.unsqueeze(1), sparse_for_mask_q, all_true)
            elif cluster == BD:
                in_window = position_diff <= W
                sparse_for_mask_q = in_window | sink_punct.unsqueeze(0)
                head_mask = torch.where(is_mask_query.unsqueeze(1), sparse_for_mask_q, all_true)
            else:
                head_mask = all_true
            head_masks.append(head_mask)
        full_mask = torch.stack(head_masks, dim=0)  # (H, T, T)

        # Apply mask: scores[..., i, j] = -inf where full_mask[h, i, j] = False
        scores = scores.masked_fill(~full_mask.unsqueeze(0), float("-inf"))

        # Also include attention_bias if provided (e.g., padding mask)
        if attention_bias is not None:
            scores = scores + attention_bias

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        out = torch.matmul(attn, v_h)  # (B, H, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.attn_out(out)

        return out, None, attn if output_attentions else None

    return _sparse_attention


def install_sparse_attention(model, tokenizer, head_policy_npy_path, mask_id=126336,
                              per_head_config_path=None):
    """Install sparse attention based on cluster assignments.

    head_policy_npy_path: (n_layers, n_heads) int8 cluster IDs
                          (use the 5-shot majority typology output).
    per_head_config_path: optional .npz with per-head (S, W) overrides
                          (config_W[layer, head] >= 0 means use that W; -1 = full)
    """
    head_clusters = np.load(head_policy_npy_path)
    n_layers, n_heads = head_clusters.shape

    # Optional per-head W (overrides cluster default)
    per_head_W = None
    if per_head_config_path is not None:
        cfg = np.load(per_head_config_path)
        per_head_W = cfg["config_W"]
        print(f"[sparse] loaded per-head W config from {per_head_config_path}")
        n_pure_window = int((per_head_W >= 0).sum())
        n_full = int((per_head_W < 0).sum())
        print(f"  per-head W>=0: {n_pure_window}  full-attention fallback: {n_full}")

    # Build (layer, head) → cluster, W maps
    head_cluster_map = {}
    head_W_map = {}
    for li in range(n_layers):
        for hi in range(n_heads):
            c = int(head_clusters[li, hi])
            head_cluster_map[(li, hi)] = c
            if per_head_W is not None:
                w = int(per_head_W[li, hi])
                if w < 0:
                    head_W_map[(li, hi)] = None  # full attention
                else:
                    head_W_map[(li, hi)] = w  # per-head W
            else:
                head_W_map[(li, hi)] = DEFAULT_W.get(c, None)

    # Build punct/special sets
    punct_ids, special_ids = build_punct_special_id_sets(tokenizer)
    print(f"[sparse] punct ids: {len(punct_ids)}  special ids: {len(special_ids)}")

    # Stats
    counts = {}
    for c in head_cluster_map.values():
        counts[c] = counts.get(c, 0) + 1
    print(f"[sparse] head cluster distribution:")
    for c, n in counts.items():
        cn = CLUSTER_NAMES[c] if 0 <= c < len(CLUSTER_NAMES) else "?"
        W = DEFAULT_W.get(c, "full")
        print(f"  {cn:>18s}: {n:>4d} heads  W={W}")

    # Patch each block
    blocks = list(model.model.transformer.blocks)
    for li, block in enumerate(blocks):
        block._layer_idx = li
        # Hook to record current input ids per forward
        def make_input_id_hook():
            def hook(module, args, kwargs):
                # First positional arg to forward is x (input_ids embeddings or tokens?)
                # For LLaDA block, input is hidden states, not tokens.
                # We need token ids from elsewhere — set via outer hook.
                pass
            return hook
        # Install the sparse attention
        block.attention = types.MethodType(
            make_sparse_attention(li, head_cluster_map, head_W_map,
                                  punct_ids, special_ids, mask_id),
            block,
        )

    # We need to store input_ids on each block per forward for sparse_attention to use.
    # Cleanest: hook on the top-level model that records input_ids, then attention reads it.
    def model_forward_hook(module, args, kwargs, output):
        return output
    # Simpler: monkey-patch the model.forward to stash input_ids on every block.
    orig_forward = model.forward
    def patched_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if input_ids is not None:
            for b in blocks:
                b._current_input_ids = input_ids
        return orig_forward(*args, **kwargs)
    model.forward = patched_forward
    print(f"[sparse] installed sparse attention on {len(blocks)} layers")
