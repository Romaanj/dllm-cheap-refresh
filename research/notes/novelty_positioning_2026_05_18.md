---
title: Novelty positioning & storyline options after lit-search (2026-05-18)
date: 2026-05-18
status: working draft
related: [[lit/spa_cache_2602_02544]], [[lit/window_diffusion_2601_20332]], [[lit/dinfer_2510_08666]], [[notes/phase12_cascade_oracle]], [[notes/phase13b_cheap_estimator]], [[notes/R26_scope_incremental_negative_result]]
---

# Headline finding

**SPA-Cache (2602.02544, Jan 2026) substantially overlaps with our
Phase 12-14 method.** Same paradigm (per-(layer × position) refresh-only),
same granularity, per-layer adaptive budget. Concurrent (5 months ahead).

Our remaining defensible novelty is narrower than originally hoped:

1. **Mechanistic diagnosis** (R26 negative + Phase 12 cascade containment + Phase 13a position geometry) — the "WHY refresh-only works for bidir attention", which SPA-Cache lacks.
2. **Simpler signal** (lag-1 history vs SVD-projected Value proxy) — easier to deploy, no offline calibration, but possibly less accurate.
3. **Structural inductive bias** (U_t + window4 explicitly included).
4. **End-to-end accuracy preservation** (GSM8K + HumanEval at baseline parity).

# Competitive landscape (concise)

| paper | granularity | signal | paradigm | speedup claim |
|---|---|---|---|---|
| Fast-dLLM (May'25) | block | block-boundary | reuse + parallel decode | 11× GSM8K |
| dKV-Cache (May'25) | per-token | delayed-after-decode | reuse | 2-10× |
| dLLM-Cache (Jun'25) | per-token | feature similarity | reuse | 5-9× |
| SparseD (Sep'25) | per-head | static-pattern | sparse attn | — |
| d²Cache (Sep'25) | per-token | drift score | reuse + selective update | — |
| Elastic-Cache (Oct'25) | per-layer | attention-drift on top-1 | reuse + refresh trigger | 8.7× GSM8K, 45.1× long-seq |
| dInfer (Oct'25) | block-vicinity | geometric | system framework | 1100 TPS HE, 10× over Fast-dLLM |
| MaskKV (Oct'25) | per-head | I/S score | reuse + prompt evict | — |
| Window-Diffusion (Jan'26) | window-uniform | geometric + phase | prune + buffer + refresh | 99× (mostly EOS-early-stop) |
| **SPA-Cache (Jan'26)** | **per-(layer × position)** | **SVD-Value proxy cos-sim** | **refresh-only** | **8× / 2-4× over cache baselines** |
| EntropyCache (Mar'26) | global | entropy of decoded tokens | reuse + refresh trigger | O(V) signal |
| **OURS (Phase 12-14)** | **per-(layer × position)** | **lag-1 hidden-state delta + U_t/window** | **refresh-only** | **oracle K/T=0.10→10× ceiling; lag-1 K/T=0.28→3.5×** |

# Three storyline options

## Storyline A — "Refresh-Only Caching: A Mechanistic Defense" (recommended)

**Framing**: 

> "Recent work (SPA-Cache, our concurrent work) proposes per-(layer ×
> position) refresh-only caching for dLLM. We complement this design choice
> with a mechanistic justification: (i) reuse-based caching fundamentally
> fails for bidirectional dLLM (R26 negative result), (ii) cascade
> containment holds for refresh-only with even simple lag-1 history
> signals, (iii) the temporal-persistence of position-level drift makes a
> nearly-free signal viable."

**Contribution chain**:
1. R26 (negative): scope-based reuse paradigm fails — diagnosis (bidir cascade r=0.86, peak mobility 57-80%, drift signal ≠ output drift).
2. Phase 12 oracle: refresh-only at K/T=0.10-0.28 achieves cos 0.998 / argmax 0.97 — cascade *contains* under refresh, not reuse.
3. Phase 13a geometry: oracle positions live mostly in prompt (50-77%) — refuting "U_t-local" intuition; need history-based signal.
4. Phase 13b cheap signal: lag-1 hidden-state delta captures 80%+ of oracle gain; nearly zero compute (one extra cached snapshot).
5. Phase 14 e2e: GSM8K and HumanEval baseline-parity at K=128-384.
6. Phase 14b (planned): masked attention/FFN kernel for actual wall-clock speedup.

**Pros**: defensible (we have unique mechanistic content), the "why" angle is missing from SPA-Cache. Lets us cite SPA-Cache as concurrent without conceding scoop.

**Cons**: weaker novelty claim; reviewers may say "we already know it works (SPA-Cache); why do we need the why".

## Storyline B — "Lag-1 History Suffices: A Free Signal for dLLM Refresh"

**Framing**:

> "Concurrent work (SPA-Cache) uses a learned SVD-projected Value-space
> proxy for selective recomputation in dLLM. We show that an even simpler
> signal — the previous step's hidden-state delta — recovers 80% of the
> oracle ceiling at near-zero compute, no offline calibration, and matches
> baseline accuracy on GSM8K + HumanEval."

**Contribution chain**:
1. Position the problem identically to SPA-Cache (refresh-only per-(layer × position)).
2. Section-by-section ablation: lag-1 history vs SPA-Cache's SVD proxy vs random vs geometric (window-only).
3. Show parity (or near-parity) with much simpler implementation.
4. End-to-end + kernel + benchmarks.

**Pros**: clean numerical comparison story. Reproducible.

**Cons**: incremental. Reviewers will ask "if the simple thing works, why didn't SPA-Cache try this first?". Requires reproducing SPA-Cache (effort).

## Storyline C — Characterization paper (no method)

**Framing**:

> "We characterize the cache-vs-refresh tradeoff in diffusion LMs through
> a sequence of oracle measurements: (i) reuse-paradigm failure modes,
> (ii) refresh-only ceiling, (iii) geometry of the oracle active set,
> (iv) temporal-persistence of drift, (v) cross-task and cross-model
> generalization."

**Contribution chain**:
1. R26 negative result (publishable mechanistic finding)
2. Phase 12-13 oracle analyses (rich phenomenology)
3. Optional small-scale method as illustration

**Pros**: zero overlap with SPA-Cache (method-free). Easier to write, faster to submit.

**Cons**: lower-impact venue (workshop, short paper). User explicitly wants a method paper + kernel work.

## Storyline D — Composition + engineering paper

**Framing**:

> "A practical dLLM inference system combining (i) per-(layer × position)
> refresh-only with lag-1 signal, (ii) masked attention/FFN CUDA kernels,
> (iii) composition with Fast-dLLM block decoding + suffix pruning. End-to-end
> X tokens/sec on GSM8K / HumanEval, beating SOTA at matched accuracy."

**Contribution chain**:
1. Phase 12-14 (method) + Phase 14b (kernel) + composition
2. Beat dInfer / Fast-dLLM / SPA-Cache on wall-clock
3. Engineering-paper venue (SOSP/OSDI? or as ML systems track at ICLR/NeurIPS)

**Pros**: highest practical impact, hard for reviewers to discount engineering wins.

**Cons**: requires substantial CUDA work (multiple weeks). Less novel-method, more system. User said they will do kernel work.

# My recommendation: A + D combined

- **Core paper**: Storyline A's mechanistic story + Storyline D's kernel + benchmarks.
- **Acknowledge SPA-Cache as concurrent**; differentiate by:
  - The mechanistic "why" (R26 + cascade containment)
  - Simpler signal (lag-1) — head-to-head ablation vs SPA-Cache
  - End-to-end system metrics including wall-clock with our kernel
- Cite SPA-Cache prominently; don't pretend it doesn't exist.

# What we should DO next to lock in this story

1. **Reproduce SPA-Cache** (or use their numbers if code released) — direct comparison numbers on same setup are essential. Without it, we can't claim "comparable accuracy with simpler signal".
2. **Phase 14b kernel** — masked attention + masked FFN for selected positions. Real wall-clock numbers.
3. **Cross-model verification** — Dream-7B (lag-1 should also work there).
4. **Cross-task expansion** — beyond GSM8K + HumanEval (MBPP, MMLU, BBH).
5. **Comparison table including dInfer** — engineering baseline.

# Honest assessment

Our work is **no longer the first** per-(layer × position) refresh-only
for dLLM. SPA-Cache scooped that framing. We are now in a position where:
- We have the **best mechanistic story** in the space.
- We have a **simpler signal** that might (need verification) match their accuracy.
- We need **kernel-level execution** to be competitive on the engineering axis.

Risk of submitting without addressing SPA-Cache directly: rejected as
"already known". Risk of pivoting to characterization paper: lower impact.
Best path is to embrace the comparison and let the simpler signal +
mechanistic depth speak.
