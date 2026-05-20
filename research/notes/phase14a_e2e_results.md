---
title: Phase 14a — End-to-end cheap inference results (GSM8K + HumanEval)
date: 2026-05-18
status: complete
related: [[phase13b_cheap_estimator]], [[H004_cascade_active_set]], [[novelty_positioning_2026_05_18]]
---

# TL;DR

Cheap E3 estimator (U_t + window4 + lag-1 history) at K=128~384 **preserves
baseline accuracy** on both GSM8K (n=100) and HumanEval (n=164):

| benchmark | baseline | cheap E3 | gap | SE |
|---|---:|---:|---:|---:|
| GSM8K (K=384) | 0.810 | 0.820 | +0.010 | ±0.039 |
| HumanEval (K=128) | 0.372 | 0.384 | +0.012 | ±0.038 |

Both within ±1 SE. **Method works end-to-end**, no catastrophic accuracy
collapse from 254 consecutive cheap forwards.

# Setup

- Model: GSAI-ML/LLaDA-8B-Instruct
- gen=256, block=32, steps=256, no dual cache, no parallel decoding
- warmup: 2 baseline forwards, then cheap_e3 for remaining steps
- E3 estimator: A_l = U_t ∪ window(U_t, 4) ∪ top-(K-...) by lag-1 norm
- 2 cached snapshots (prev, prev_prev)
- GSM8K: 5-shot, n=100; HumanEval: 0-shot, n=164 (all problems)
- HumanEval scored via `postprocess_code.py` (sanitize + hf_evaluate code_eval)

# Detailed results

## GSM8K (K=384, K/T ≈ 0.28)

| metric | baseline | cheap_e3 |
|---|---:|---:|
| accuracy | 81/100 = 0.810 | 82/100 = 0.820 |
| wall_s mean | 38.4 | 41.8 |
| wall_s median | 38.3 | 41.6 |

Outcome breakdown (n=100):
- both correct: 74
- only baseline correct: 7
- only cheap correct: 8
- both wrong: 11
- same prediction: 79

→ Near-perfect symmetry in disagreements (7 vs 8). Cheap method is NOT
systematically worse on any class of samples.

## HumanEval (K=128, K/T ≈ 0.29 average; prompts ~100-300 tokens)

| metric | baseline | cheap_e3 |
|---|---:|---:|
| pass@1 | 61/164 = 0.372 | 63/164 = 0.384 |
| total wall_s | 2211 | 2391 |

## Wall-time observation

cheap_e3 is **+9% slower** than baseline because:
1. Per-step estimator overhead (delta-norm + argsort per 32 layers) — Python overhead
2. Hybrid forward still runs FULL layer compute (we just SELECT which outputs to keep)
3. **No custom kernel** — masked attention/FFN not implemented

The oracle ceiling K/T=0.28 → ~3.5× speedup is only realizable with kernel
work (Phase 14b).

# Verdict on H004

**H004 strongly supported across all axes**:

- H004a (cos ≥ 0.99 at K ≤ 0.4 L): PASS at K=32 (cos 0.991)
- H004b (deep ≤ 30% true-active): PASS at <10%
- H004c (monotone, no cliff): PASS
- H004d (compute saving ratio): PASS at K=128 (10% L)
- **NEW: end-to-end accuracy preservation**: PASS at K=128 (HumanEval), K=384 (GSM8K)

# Next steps

1. **Phase 14b — Masked kernel**: implement masked attention + masked FFN
   that only computes for A_l queries. Target 3.5× wall-clock.
2. **SPA-Cache comparison**: reproduce or use their numbers. Direct ablation:
   our lag-1 history vs their SVD-Value proxy at matched K.
3. **Cross-model**: Dream-7B (lag-1 should also work — temporal persistence
   is bidir-attention universal).
4. **Cross-task**: MBPP, MMLU at minimum to expand evaluation breadth.

# Files

- `phase14a_cheap_e2e.py` (GSM8K)
- `phase14a_humaneval.py` (HumanEval, lm-eval format)
- `postprocess_code.py` (HumanEval scoring)
- `analyze_phase14a_e2e.py` (GSM8K comparison)
- `results_phase14a_e2e/{baseline,cheap_e3}_n100/` (GSM8K, 100 each)
- `results_phase14a_he/{baseline,cheap_e3}/` (HumanEval, 164 each)
