"""SCOPE-DRIVEN INCREMENTAL attention for LLaDA-8B dLLM.

Implements per-(layer, head, query) attention-output cache with
scope-based incremental updates. Per inference step:
  - U_k = positions whose token_ids changed since prev step (typically 1-2)
  - For each (layer, head, query):
      cluster = head_cluster[layer, head]
      affected = (scope(cluster, query, W_h) ∩ U_k ≠ ∅)
      if affected: compute new attention output
      else: reuse cached output from prev step
  - Update cache for next step.

V1 design (correctness-first):
  - Full attention forward is still computed (no compute saving yet).
  - Cache replaces attention output at unaffected (head, query) cells.
  - Validates that scope-based freezing preserves accuracy.

V2 (later): skip attention computation for unaffected cells (real
compute saving).
"""

import math
import types
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
FRONTIER = CLUSTER_NAMES.index("frontier_tracker")
STAT_SINK = CLUSTER_NAMES.index("stationary_sink")

# Window per cluster.
# V1: 10/5/150  (E3 N=50 = 0.48 — too tight, cascading error)
# V2: 32/16/300 (sparse_v3 sweet spot — try first)
DEFAULT_W = {
    MB: 32,
    PS: 16,
    BD: 300,
}


def apply_rotary_at_positions(rotary_emb, x, positions):
    """Apply rotary to x (..., T, hd) at given absolute positions."""
    seq_len = rotary_emb.config.max_sequence_length
    pos_sin, pos_cos = rotary_emb.get_rotary_embedding(seq_len, x.device)
    pos_sin = pos_sin.type_as(x)
    pos_cos = pos_cos.type_as(x)
    pos_sin_sel = pos_sin.index_select(2, positions)
    pos_cos_sel = pos_cos.index_select(2, positions)
    return rotary_emb.apply_rotary_pos_emb(pos_sin_sel, pos_cos_sel, x)


def build_punct_special_id_sets(tokenizer):
    """Punct + special token IDs (used to identify punct positions in sequence)."""
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


class ScopeIncrementalCache:
    """Per-(layer, head, query) attention output cache + change tracking."""

    def __init__(self, n_layers, n_heads, head_clusters, head_W_map, mask_id):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_clusters = head_clusters  # (L, H) numpy int
        self.head_W_map = head_W_map        # dict cluster_id → W (int)
        self.mask_id = mask_id
        self.attn_out_cache = {}            # layer_idx → (B, H, T, hd) tensor
        self.prev_token_ids = None
        self.last_T = None
        self.reuse_stats = {"affected": 0, "reused": 0}

    def reset(self):
        self.attn_out_cache = {}
        self.prev_token_ids = None
        self.last_T = None

    def is_new_generation(self, current_token_ids):
        """Heuristic: new generation if T differs OR if no prev tokens."""
        T = current_token_ids.shape[-1]
        if self.last_T is None or self.last_T != T:
            return True
        return False

    def compute_U_k(self, current_token_ids):
        """Return tensor of positions where tokens changed from prev step.
        None if prefill (no prev state)."""
        if self.prev_token_ids is None:
            return None
        if self.prev_token_ids.shape != current_token_ids.shape:
            return None
        diff_mask = (current_token_ids != self.prev_token_ids)
        if not diff_mask.any():
            return torch.tensor([], dtype=torch.long, device=current_token_ids.device)
        return diff_mask.nonzero(as_tuple=True)[0]

    def compute_affected_mask(self, layer_idx, T, U_k, punct_pos_mask, device):
        """Returns (n_heads, T) bool — head h's query q is affected by U_k.

        punct_pos_mask: (T,) bool — which positions hold a punct/special token.
        """
        if U_k is None or U_k.numel() == 0:
            return torch.zeros(self.n_heads, T, dtype=torch.bool, device=device)

        positions = torch.arange(T, device=device, dtype=torch.long)
        # Min distance from each query to any U_k pos: (T,)
        distances = (positions.unsqueeze(1) - U_k.unsqueeze(0)).abs()  # (T, n_U)
        min_dist = distances.min(dim=1).values

        # Did any U_k position have a punct token?
        any_U_k_punct = bool(punct_pos_mask[U_k].any().item()) if punct_pos_mask is not None else False

        # Per-head decision
        clusters_layer = self.head_clusters[layer_idx]  # numpy (H,)
        affected = torch.zeros(self.n_heads, T, dtype=torch.bool, device=device)
        true_T = torch.ones(T, dtype=torch.bool, device=device)
        false_T = torch.zeros(T, dtype=torch.bool, device=device)

        for h in range(self.n_heads):
            c = int(clusters_layer[h])
            W = self.head_W_map.get(c, None)
            if W is None:
                affected[h] = true_T  # diffuse / other → always recompute
            elif c == MB:
                affected[h] = (min_dist <= W)
            elif c == PS:
                affected[h] = true_T.fill_(any_U_k_punct)
            elif c == BD:
                affected[h] = (min_dist <= W) | (true_T if any_U_k_punct else false_T)
            else:
                affected[h] = true_T  # unknown cluster → safe fallback

        return affected

    def gate(self, layer_idx, attn_output, affected_mask):
        """Combine new attn_output with cached. Update cache."""
        cached = self.attn_out_cache.get(layer_idx)
        if cached is None or cached.shape != attn_output.shape:
            self.attn_out_cache[layer_idx] = attn_output.detach().clone()
            return attn_output
        # affected: (H, T). Broadcast to (1, H, T, 1).
        mask = affected_mask.unsqueeze(0).unsqueeze(-1)
        new_out = torch.where(mask, attn_output, cached)
        self.attn_out_cache[layer_idx] = new_out.detach().clone()
        # Stats
        self.reuse_stats["affected"] += int(affected_mask.sum().item())
        self.reuse_stats["reused"] += int((~affected_mask).sum().item())
        return new_out

    def commit_tokens(self, current_token_ids):
        """Save current_token_ids as the new prev_token_ids for next step."""
        self.prev_token_ids = current_token_ids.detach().clone()
        self.last_T = current_token_ids.shape[-1]


def make_patched_attention(layer_idx):
    """Return a per-block attention method that uses scope cache."""
    def _attn(self, q, k, v, mask=None, attention_bias=None,
              layer_past=None, use_cache=False,
              replace_position=None, output_attentions=False):
        if attention_bias is None and mask is not None:
            attention_bias = mask
            mask = None

        cache = self._scope_cache
        B, T, C = q.size()
        n_heads = self.config.n_heads
        head_dim = C // n_heads
        scale = 1.0 / math.sqrt(head_dim)

        # Norms
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=k.dtype)
            k = self.k_norm(k).to(dtype=k.dtype)

        # (B, T, C) -> (B, H, T, hd)
        Q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        K = k.view(B, T, n_heads, head_dim).transpose(1, 2)
        V = v.view(B, T, n_heads, head_dim).transpose(1, 2)

        # Rotary
        positions = torch.arange(T, device=q.device, dtype=torch.long)
        Q = apply_rotary_at_positions(self.rotary_emb, Q, positions)
        K = apply_rotary_at_positions(self.rotary_emb, K, positions)

        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if attention_bias is not None:
            scores = scores + attention_bias
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn, V)  # (B, H, T, hd)

        # Apply scope incremental cache
        token_ids = self._current_input_ids[0]  # (T,)
        U_k = cache.compute_U_k(token_ids)
        if U_k is None:
            # prefill — save and use all
            cache.attn_out_cache[layer_idx] = output.detach().clone()
            layer_output = output
        else:
            punct_pos_mask = self._current_punct_pos_mask
            affected = cache.compute_affected_mask(
                layer_idx, T, U_k, punct_pos_mask, q.device,
            )
            layer_output = cache.gate(layer_idx, output, affected)

        # Concat heads + project
        layer_output = layer_output.transpose(1, 2).contiguous().view(B, T, C)
        layer_output = self.attn_out(layer_output)
        return layer_output, None, attn if output_attentions else None

    return _attn


def install_scope_incremental(model, tokenizer, head_policy_npy_path,
                              mask_id=126336, head_W_overrides=None):
    """Patch model to use scope-driven incremental attention.

    Returns the cache object so caller can reset between generations.
    """
    head_clusters = np.load(head_policy_npy_path).astype(np.int8)
    n_layers, n_heads = head_clusters.shape

    head_W_map = dict(DEFAULT_W)
    if head_W_overrides:
        head_W_map.update(head_W_overrides)

    cache = ScopeIncrementalCache(
        n_layers=n_layers, n_heads=n_heads,
        head_clusters=head_clusters, head_W_map=head_W_map, mask_id=mask_id,
    )

    # Token-id classification (precomputed once)
    punct_ids, special_ids = build_punct_special_id_sets(tokenizer)
    punct_ids_tensor = torch.as_tensor(sorted(punct_ids | special_ids),
                                        dtype=torch.long)

    # Cluster distribution log
    counts = {}
    for h_row in head_clusters:
        for c in h_row:
            counts[int(c)] = counts.get(int(c), 0) + 1
    print(f"[scope] head cluster distribution:")
    for c, n in counts.items():
        cn = CLUSTER_NAMES[c] if 0 <= c < len(CLUSTER_NAMES) else "?"
        W = head_W_map.get(c, "full")
        print(f"  {cn:>18s}: {n:>4d} heads  W={W}")
    print(f"[scope] punct/special ids: {len(punct_ids)}+{len(special_ids)}")

    blocks = list(model.model.transformer.blocks)
    for li, block in enumerate(blocks):
        block._scope_cache = cache
        block._layer_idx = li
        block.attention = types.MethodType(make_patched_attention(li), block)

    # Top-level forward wrapper: stash token_ids and punct mask on each block
    orig_forward = model.forward
    punct_ids_set = punct_ids | special_ids

    def patched_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if input_ids is not None:
            tokens = input_ids[0]
            # Compute punct position mask
            device = tokens.device
            pids = punct_ids_tensor.to(device)
            punct_pos_mask = torch.isin(tokens, pids)
            # Auto-reset cache if T changed (new generation)
            if cache.is_new_generation(tokens):
                cache.reset()
            for b in blocks:
                b._current_input_ids = input_ids
                b._current_punct_pos_mask = punct_pos_mask
            ret = orig_forward(*args, **kwargs)
            # commit after forward
            cache.commit_tokens(tokens)
            return ret
        return orig_forward(*args, **kwargs)

    model.forward = patched_forward
    print(f"[scope] installed on {len(blocks)} layers")
    return cache


def reset_cache(model):
    """Public API: reset cache (call between generations if needed)."""
    blocks = list(model.model.transformer.blocks)
    if blocks and hasattr(blocks[0], "_scope_cache"):
        blocks[0]._scope_cache.reset()
