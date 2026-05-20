# RetroAttention — Retrospective Sparse Attention for Efficient Long-Context Generation (2508.09001)

- arXiv: https://arxiv.org/abs/2508.09001
- HTML:  https://arxiv.org/html/2508.09001v1
- Setting: **autoregressive** long-context decoding (NOT dLLM)

## TL;DR

Lightweight **attention-output cache** (shape `[w-1, B, L, D]`) that
stores per-Query attention outputs from recent steps. When new KV entries
arrive in subsequent decoding steps, retrospectively update prior Queries'
attention outputs using the new K/V — a kind of "lazy attention
extension". Per-head page-loaded mask `A` records the most recent step a
KV page was loaded for each head, gating duplicate updates.

## Why it matters for us

This is **the closest analog to our planned design**, but in AR setting:

| dimension                | RetroAttention                                  | ours (dLLM)                                            |
|--------------------------|-------------------------------------------------|--------------------------------------------------------|
| cache target             | attention OUTPUT                                | attention OUTPUT                                       |
| update trigger           | new KV pages arrive each AR step                | newly-unmasked positions U_k each diffusion step       |
| gating signal            | per-(head, page) last-loaded step               | per-(head, query) scope ∩ U_k                          |
| scope concept            | retrospective window w (recency)                | empirical sink_set ∪ ±window per head                  |
| compute saving           | avoid full-context attention for past queries   | skip full attention for unaffected (head, query)       |
| setting                  | AR causal                                       | bidirectional masked-diffusion                         |

**Same skeleton; different domain & different scope source.** RetroAttention
proves the attention-output-cache idea is sound in AR. Our novelty is:
- Bidirectional (dLLM) setting where reuse is harder because newly-arrived
  positions are arbitrary in sequence-position, not strictly future.
- Scope from observation (mask_binder / punct_sink / bidir_asym_L) rather
  than recency window.
- Set-disjointness criterion (`scope ∩ U_k = ∅`), not last-step-loaded.

## Citation

```
@article{retroattention_2508_09001,
  title  = {Retrospective Sparse Attention for Efficient Long-Context Generation},
  year   = {2025}
}
```

Cross-links: [[d2cache_2509_23094]] [[entropycache_2603_18489]]
