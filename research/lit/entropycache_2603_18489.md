# EntropyCache — Decoded Token Entropy Guided KV Caching for dLLMs (2603.18489)

- arXiv: https://arxiv.org/abs/2603.18489
- HTML:  https://arxiv.org/html/2603.18489

## TL;DR

After each unmasking step, evaluate max-entropy of newly-decoded token
distributions. If entropy > threshold → full forward pass; else reuse
cached KV for most positions and **recompute only current mask tokens +
the k most-recently unmasked tokens** (k=64).

## Mechanism

- Cache type: **KV cache** (not attention output).
- Granularity: **per-layer per-position**. At each layer, Q/K/V are
  recomputed only for the index set `M_t ∪ R_t` where
    - `M_t` = currently masked positions
    - `R_t` = k positions with most recent decode history
- Trigger: entropy of newly-unmasked logits (global signal).
- No per-head scope. No mask_binder / punct_sink / bidir_asym
  distinction.

## Relation to our scope-driven idea

Closest of all surveyed papers to "event-driven recompute on newly
unmasked": `R_t` is a recency-truncated proxy for "tokens that just
changed". But:

| dimension          | EntropyCache                                | ours                                                |
|--------------------|---------------------------------------------|-----------------------------------------------------|
| affected positions | recency rank, top-k=64 globally             | per-head: scope(head, q) ∩ U_k                       |
| per-head?          | no                                          | yes — heads differ in window radius and sink set     |
| dependency model   | recency proxy                               | empirical (probe-derived) scope                      |
| cache              | KV                                          | attention output                                     |
| invalidation rule  | global entropy threshold                    | set-disjointness per (layer, head, query)            |

**Where we differ structurally:** EntropyCache assumes one global k for
all heads/layers; we let head h₁ (mask_binder, ±5) refresh on ~5 queries
while head h₂ (punct_sink) refreshes only if a punct position flips.
That's a strict refinement, not a re-derivation.

## Citation

```
@article{entropycache_2603_18489,
  title  = {EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models},
  year   = {2026}
}
```

Cross-links: [[d2cache_2509_23094]] [[elastic_cache_2510_14973]]
[[retroattention_2508_09001]]
