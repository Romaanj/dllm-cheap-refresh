---
title: Phase 13b — Cheap (lag-1) A_l estimator vs oracle ceiling
date: 2026-05-18
status: complete
related: [[H004_cascade_active_set]], [[phase12_cascade_oracle]], [[phase13a_oracle_geometry]]
---

# TL;DR

**Cheap lag-1-history estimator captures 80%+ of oracle gain.** At
K=384 (28% L operating point):
- Oracle argmax: 0.966
- E1 lag-1 argmax: **0.947** (oracle gap 1.9pp)
- E3 U_t+window4+lag-1: 0.952 (oracle gap 1.4pp)
- Random argmax: 0.865 (baseline / "no info")

Translation: lag-1 estimator at K=384 ≈ oracle at K=256. **Cheap-vs-oracle
penalty = ~1.5× more refresh budget** for matching accuracy. The
estimator requires zero extra forward pass — only one extra cached
snapshot (prev_prev_block_out).

→ **Cheap-method route is viable.** Phase 13 question (can we approximate
oracle cheaply?) answered YES.

# Setup

- Script: `phase13b_cheap_estimator.py`
- 8 GSM8K test samples, 5-shot, gen=256, block=32, probe stride 16
  → 120 probe steps total
- K grid: {64, 128, 256, 384, 512}
- 5 estimators × 5 K = 25 hybrid forwards per probe = 3000 rows

## Estimators tested

| Name | A_l definition | Cost |
|---|---|---|
| **E0_oracle** | top-K by `||cur_block_out[l] − prev_block_out[l]||` | Cheating (needs cur) — Phase 12 reference |
| **E1_lag1** | top-K by `||prev_block_out[l] − prev_prev_block_out[l]||` | ~Free (one extra snapshot) |
| **E2_u_lag1** | U_t ∪ top-(K−|U|) by lag-1 | ~Free |
| **E3_uw_lag1** | U_t ∪ window(U_t, 4) ∪ top-(...) by lag-1 | ~Free |
| **E4_random** | random K positions | Sanity baseline |

# Results

## Argmax-agreement per (estimator, K)

| K | K/T | E0 oracle | E1 lag-1 | E3 u+w+lag | E4 random |
|---:|---:|---:|---:|---:|---:|
| 64  | 0.05 | 0.899 | 0.873 | 0.888 | 0.865 |
| 128 | 0.10 | 0.929 | 0.888 | 0.897 | 0.865 |
| 256 | 0.19 | 0.945 | 0.931 | 0.939 | 0.866 |
| 384 | 0.28 | **0.966** | **0.947** | **0.952** | 0.865 |
| 512 | 0.38 | 0.969 | 0.957 | 0.957 | 0.866 |

## Cos-logits per (estimator, K)

| K | E0 oracle | E1 lag-1 | E3 u+w+lag | E4 random |
|---:|---:|---:|---:|---:|
| 128 | 0.9980 | 0.9912 | 0.9957 | 0.9854 |
| 256 | 0.9991 | 0.9957 | 0.9981 | 0.9853 |
| 384 | 0.9995 | 0.9975 | 0.9990 | 0.9853 |
| 512 | 0.9997 | 0.9985 | 0.9993 | 0.9853 |

## Gain capture vs oracle (relative to random baseline)

| K | oracle gain | lag-1 gain | lag-1 / oracle |
|---:|---:|---:|---:|
| 64  | 0.034 | 0.008 | 23% |
| 128 | 0.064 | 0.023 | 36% |
| 256 | 0.080 | 0.065 | 82% |
| 384 | 0.101 | 0.082 | **81%** |
| 512 | 0.103 | 0.091 | **88%** |

→ Lag-1 captures 80%+ at K≥256. Lower-K regime (64-128) is harder —
cheap estimator can't recover the small but precise oracle picks.

## IoU(estimator, oracle) at layer 15 (representative deep-ish)

| estimator | K=128 | K=256 | K=384 | K=512 |
|---|---:|---:|---:|---:|
| E1 lag-1 | 0.43 | 0.49 | 0.52 | 0.54 |
| E3 u+w+lag | 0.44 | 0.49 | 0.52 | 0.54 |
| E4 random | 0.05 | 0.10 | 0.17 | 0.23 |

Lag-1 IoU is 4-9× higher than random, but absolute value stays around 0.5.
**Yet despite IoU only ≈ 0.5, lag-1 achieves 80%+ of oracle's argmax
gain** — meaning the positions lag-1 *misses* are largely interchangeable
with positions lag-1 *picks*. Phase 12's cascade containment absorbs the
mis-picks.

## E1 vs E2 vs E3 (incremental contribution of U_t, window)

- E1 lag-1 alone: argmax 0.947 at K=384
- E2 = E1 + U_t (free positions): 0.948 — **adding U_t alone: +0.1pp**
- E3 = E2 + window(U_t, 4): 0.952 — **adding window4: +0.4pp**

U_t is already implicit in lag-1 (it was the trigger of the previous
step's deltas, so already near the top of lag-1 ordering). Window adds
small but consistent benefit, mainly at smaller K.

# Implications

## Method-design viability — YES

- Lag-1 history is the right cheap signal.
- Memory cost: one extra `block_out` snapshot per layer ≈ 345 MB at L=1316.
  Compatible with current LLaDA-8B setup.
- Compute cost: per-step per-layer delta norm + argsort = trivial (<1ms total).

## Operating-point recommendation

For target argmax ≥ 0.95:
- Oracle: K=384 (28% L)
- Cheap (E3): K=384 (28% L) — argmax 0.952, just above threshold
- Effective compute ratio: **0.28** → **3.5× attention/FFN speedup ceiling**

For target argmax ≥ 0.93:
- Cheap (E3): K=256 (19% L) — argmax 0.939
- Effective compute ratio: 0.19 → **5.3× ceiling**

These are oracle-style measurements; real wall-clock requires masked
attention/FFN implementation that actually skips non-A_l queries.

## What's still missing for a paper

1. **Wall-clock vs FLOPS-counted speedup**: actual masked-attention
   kernel that skips ~A_l rows. Requires CUDA work; current oracle just
   counts positions.
2. **End-to-end GSM8K accuracy**: argmax 0.95 ≠ accuracy 0.95. Need full
   decode + eval with cheap estimator on full 1319.
3. **Per-layer adaptive K**: deeper layers have more active positions
   (Phase 12 finding), shallower have less; layer-stratified K could
   close the cheap-vs-oracle gap.
4. **Composition with Fast-dLLM block-decode + Q1a per-head sparse**:
   orthogonality test.

# Decision

Recommended next phase: **Phase 14 — End-to-end GSM8K eval with cheap
estimator + masked attention**. This is the gap between "oracle ceiling
result" and "method works in practice."

Open question: implement masked attention now (CUDA-ish work), or first
measure GSM8K accuracy with the oracle counter (no kernel work)?
The latter is cheaper and tells us whether the argmax→accuracy translation
holds.

# Files

- `phase13b_cheap_estimator.py`
- `analyze_phase13b_cheap.py`
- `results_phase13b_cheap/gpu{0,1}/cheap_estimator.jsonl` (3000 rows)
- `analysis_phase13b_cheap.txt`
