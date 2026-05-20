---
title: Phase 13a — Oracle active-set geometry (where do active positions live?)
date: 2026-05-18
status: complete
related: [[H004_cascade_active_set]], [[phase12_cascade_oracle]], [[phase13b_cheap_estimator]]
---

# TL;DR

**The Phase 12 oracle active-set is NOT a simple geometric function of U_t
(newly unmasked positions).** Median distance from oracle position to
nearest U_t is 150-450 across all layers. At K=384, even the best
geometric estimator (U_t ∪ window(w=128)) achieves only IoU 0.20-0.41
with oracle.

More striking: **50-77% of oracle positions are in the PROMPT region**
(not the generation region) — bidir attention propagates U_t's effect to
prompt hidden states even though prompt tokens never change.

→ Single-step geometric proxy is insufficient. **History-based (lag-1)
estimator is required** — see [[phase13b_cheap_estimator]] for end-to-end
test.

# Setup

- Script: `phase13a_capture_active_set.py`
- 8 GSM8K test samples, 5-shot, gen=256, block=32, steps=256 (stride 1)
- Per probe step (2040 total): save U_t (positions decoded at prev step),
  top-512 oracle positions per layer by `||cur_block_out[l] − prev_block_out[l]||`,
  block boundaries, prompt_len

# Results

## Q1 — Distance from A_oracle to nearest U_t (K=384)

| layer | median | p25 | p75 | mean | f@w≤4 | f@w≤16 | f@w≤64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0  | 196 | 60  | 709 | 386 | 0.02 | 0.08 | 0.26 |
| 8  | 158 | 56  | 591 | 343 | 0.02 | 0.08 | 0.28 |
| 16 | 155 | 58  | 450 | 301 | 0.02 | 0.08 | 0.27 |
| 24 | 316 | 83  | 707 | 415 | 0.02 | 0.07 | 0.21 |
| 31 | 408 | 118 | 765 | 469 | 0.02 | 0.06 | 0.17 |

→ Oracle positions are **NOT local to U_t**. Only 2-8% of oracle within
distance 4 (even at shallow layers). Median 150-400 positions away.

## Q2 — Window-coverage(A_oracle, U_t ∪ window(U_t, w))

What fraction of A_oracle is captured by a window of width w around U_t?
At K=384:

| layer | w=0 | w=4 | w=16 | w=64 | w=128 |
|---:|---:|---:|---:|---:|---:|
| 0  | 0.00 | 0.02 | 0.08 | 0.26 | 0.41 |
| 12 | 0.00 | 0.02 | 0.08 | 0.28 | 0.46 |
| 16 | 0.00 | 0.02 | 0.08 | 0.27 | 0.45 |
| 24 | 0.00 | 0.02 | 0.07 | 0.21 | 0.33 |
| 31 | 0.00 | 0.02 | 0.06 | 0.17 | 0.26 |

→ Best coverage 41-46% at w=128. **No layer achieves ≥80% coverage** with
any window we tested. Geometric proxy too narrow.

## Q3 — IoU(A_oracle_K, U_t ∪ window(U_t, w))

At K=384, best IoU across (layer, w):

| layer | best IoU | at w |
|---:|---:|---:|
| 0  | 0.35 | 128 |
| 12 | 0.41 | 128 |
| 16 | 0.39 | 128 |
| 24 | 0.27 | 128 |
| 31 | 0.20 | 128 |

→ IoU < 0.5 at every layer with any window. Single-step geometric estimator
will not match oracle by IoU.

## Q4 — Position-type split of A_oracle

At K=384, fraction in {current block, prompt, past gen, future blocks}:

| layer | in_block | in_prompt | in_past_gen | in_future |
|---:|---:|---:|---:|---:|
| 0  | 0.08 | **0.57** | 0.16 | 0.19 |
| 16 | 0.08 | **0.51** | 0.19 | 0.22 |
| 31 | 0.06 | **0.73** | 0.07 | 0.15 |

At K=128, prompt fraction at layer 31 reaches **0.77**. Block share is
mildly over-represented (3-7× vs uniform rate), but absolute majority of
oracle lives in prompt.

# Mechanism

Bidir attention propagates U_t's effect globally. Prompt positions don't
change tokens but their hidden states drift via attention to the
now-different gen region. Since prompt is 80% of L (1060/1316), it
absorbs most of the cascade by quantity. The *which prompt positions*
question is content-dependent (likely sinks / mask_binder targets / etc.) —
not geometric.

# Implication for cheap estimator design

Geometric "U_t + window" too narrow → tested in [[phase13b_cheap_estimator]].
Need history-based signal: positions that were active last step are
likely active this step (temporal persistence).

# Files

- `phase13a_capture_active_set.py`
- `analyze_phase13a_geometry.py`
- `results_phase13a_geometry/gpu{0,1}/oracle_positions.jsonl` (2040 rows)
- `analysis_phase13a_geometry.txt`
