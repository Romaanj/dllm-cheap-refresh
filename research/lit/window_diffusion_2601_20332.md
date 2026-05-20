# Window-Diffusion: Accelerating Diffusion Language Model Inference with Windowed Token Pruning and Caching

- authors: Fengrui Zuo, Zhiwei Ke, Yiming Liu, Wenqi Lou, Chao Wang, Xuehai Zhou
- venue: arXiv preprint
- year: 2026 (submitted 2026-01-28)
- url: https://arxiv.org/abs/2601.20332
- tags: [dllm-cache, geometric-window, phase-based-refresh, suffix-pruning]

## Key claims

- Three-category tokens: **active** (computed online), **buffer** (cached + periodic refresh), **far-field** (pruned).
- Window slides rightward as denoising progresses. Window widths fixed (external=128, internal=16), uniform across layers.
- Phase-based refresh: full forward pass at the start of each phase (default 32 steps).
- Up to **99× speedup**: 6.6× from token pruning + 15× from adaptive early-stop on `<eos>`.

## Method

Sliding window over sequence positions: any token outside the external window is excluded from computation entirely. Within the window, "buffer" tokens use cached KV that get periodically (every 32 steps) refreshed via a full forward. Active tokens are at the decoding frontier.

## Why it matters here

**Window-based and uniform across layers** — different granularity from
ours (per-layer × per-position). The 99× number comes from pure
suffix-dropping + early-stop, NOT from sparse-position refresh. Their
caching design within the window is **phase-based, not per-step
drift-aware**.

| dimension | Window-Diffusion | Ours |
|---|---|---|
| granularity | per-token, **same across layers** | per-(layer × position) |
| selection signal | geometric (distance from frontier) | history-based (lag-1 delta) |
| refresh timing | every 32 steps (phase boundary) | every step |
| paradigm | active + cached + pruned | active + reused-stale |
| speedup source | suffix pruning + early stop | hypothetical attn/FFN compute reduction (oracle ceiling) |

Their gains are dominated by suffix pruning + EOS-early-stop, not by clever
within-window caching. We attack a different axis (per-layer × position
refresh) that COULD be orthogonal and composable.

## Open questions raised

- Could our per-(layer × position) refresh be composed with their
  suffix pruning for stacked gains?
- Their "buffer + periodic refresh" within window — phase=32 step refresh is
  basically Fast-dLLM's block-warm-pass.
