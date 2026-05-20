# Attention Is All You Need for KV Cache in Diffusion LLMs (Elastic-Cache)

- **arxiv**: 2510.14973 (Oct 2025)
- **URL**: https://arxiv.org/abs/2510.14973
- **Authors**: Quan Nguyen-Tri, Mukul Ranjan, Zhiqiang Shen

## Summary

Training-free adaptive KV refresh for dLLMs. Decides per-layer when to refresh
KV using attention-drift tests. Up to 45× speedup.

## Key claims relevant to our work

1. **Claim**: "distant MASK tokens primarily act as a **length-bias**" — i.e.,
   mask K is largely positional padding once block boundaries are set. This is
   the dual of our finding: they argue mask K is *low-information* on
   average, we find a 20% subpopulation of heads for which mask K is
   *critical*.
2. Uses attention drift (cos-sim between consecutive steps) as a refresh
   trigger — head-agnostic at the analysis level.
3. **What they do NOT do**: per-head clustering, identifying a mask-binding
   subpopulation, cross-task head-IoU. Their analysis is layer-aggregate.

## Overlap with our `mask_binder` finding

- **Tension, not overlap**: their average-case framing ("mask = length-bias")
  is *contradicted* on the 20% mask_binder subpopulation. This is actually a
  good framing for our contribution: "global averages hide a head-resolved
  subpopulation."
- They give us a target to position against: "Elastic-Cache treats mask K
  uniformly across heads; we show 20% of heads break this assumption."
