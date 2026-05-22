# KV drift measurement on baseline decoding — design rationale for ZAR

**Date**: 2026-05-22
**Trigger**: After v4 (window-only) ablation showed task-divergent behavior
(GSM8K strict +7.4pp vs v3, HumanEval cleaned −12.2pp), we wanted direct
empirical evidence of K/V staleness patterns to inform a better refresh
design than either Fast-dLLM's dense block-boundary refresh or v4's
single-shot window-only refresh.

## Setup

- Script: `analyze_kv_drift.py`
- Hooks: forward hooks on every block's `k_proj` and `v_proj` modules
- Decode: vanilla LLaDA full-forward (no caching) with threshold=0.9 parallel
  decoding plus force-progress (mimics our eval setup)
- Metric: per-(step, layer, position-class) drift
  - `drift_from_warmup` = 1 − cos(K_t, K_0)
  - `drift_lag1` = 1 − cos(K_t, K_{t−1})
- Tasks: HumanEval 0-shot (n=5, Lp≈100-140), GSM8K 5-shot (n=5, Lp≈1200-1290)
- gen=256, steps=256, block=32, threshold=0.9

## Position classes

Coarse:
- `prompt`        — all p ∈ [0, Lp)
- `gen_unmasked`  — currently-decoded gen positions
- `gen_masked`    — still-masked gen positions

Fine (prompt sub-bins by distance from gen boundary):
- `prompt_d0_32`     — closest 32 prompt positions (next to gen)
- `prompt_d32_128`
- `prompt_d128_512`
- `prompt_d512_plus` — far back (only meaningful for GSM8K)

## Result 1 — temporal pattern (drift_from_warmup, deep layers)

HE prompt K drift_from_warmup (deep layers 22-31):
- step 1: 0.040    step 10: 0.086   step 20: 0.105   step 50: 0.124   step 80: 0.139

GSM8K prompt K drift_from_warmup (deep):
- step 1: 0.017    step 10: 0.037   step 20: 0.044   step 50: 0.058   step 99: 0.109

Both curves show a clear "knee" around **step 20-30** (≈ 100-150 unmasked
gen tokens at our threshold).  After the knee, drift continues to climb,
but slowly.  This is NOT a true asymptote, but the per-step lag-1 drift
drops from ~0.04 (K, step 1) to ~0.01-0.02 (steady state from step 2).

GSM8K drifts about **half as fast as HE** (per step) and reaches **half the
total drift** by step 50 — consistent with the much longer prompt diluting
the per-position perturbation from each new unmasked gen token.

## Result 2 — spatial pattern (distance from gen boundary)

GSM8K prompt K drift_from_warmup at step 50:

| bin                  | drift |
|----------------------|-------|
| `prompt_d0_32`       | 0.106 |
| `prompt_d32_128`     | 0.067 |
| `prompt_d128_512`    | 0.070 |
| `prompt_d512_plus`   | 0.049 |

**Closest-to-gen prompt positions drift 2× more than the far-prompt
positions.**  d0_32 dominates throughout; d512+ stays much lower.

In HE this is weaker (Lp ≈ 135 so d512+ doesn't exist; the gap between
d0_32 and d128_512 is only ~0.05-0.07 by step 50).

## Result 3 — gen_unmasked drift dominates

`gen_unmasked` deep K drift_from_warmup reaches **0.3–0.5+** by mid-decode,
much larger than prompt's 0.1.  This is by definition — those positions'
inputs (token embeddings) changed when they were unmasked.  Lag-1 drift
for `gen_unmasked` spikes at step 1 (0.25) then drops fast to 0.02-0.04.

This is what our existing v3/v4 already handle (W_idx covers U_t).

## Three-zone implication for refresh design

| Zone | Region | Drift behavior | Strategy |
|---|---|---|---|
| A   | last ~32 prompt tokens + current block + small lookahead | high | active refresh every cheap step |
| B   | prompt[0, Lp-32) (most of prompt) | low (especially d512+) | freeze after warmup |
| C   | trailing committed gen | moderate, monotone after first jump | freeze after window-pass (v4 behavior) |

Plus temporal: extending warmup until step ~7-10 (= first block decoded)
catches the knee and reduces residual stale-from-MASK error in prompt-side
deeper layers.

## Method spec — ZAR (Zone-Aware Refresh)

Implemented in:
- `phase14a_cheap_e2e.build_zar_A` — constructs A_l = current_block ∪
  [block_end, block_end + α) ∪ [Lp - β, Lp), same across all layers
- `cheap_e3_inference_llada.generate_cheap_e3` — new `zar_mode` branch
- `eval_llada.py` — `cheap_zar_mode`, `cheap_zar_lookahead`,
  `cheap_zar_prompt_tail` model_args

Defaults: α=8, β=32, warmup=8 (extended).

Smoke test on HE 5-sample: ZAR runs cleanly, 87 NFE/sample vs v4's ~127
(extended warmup amortized; cheap-mode terminates earlier because more
refreshed positions → faster confidence build-up).

## In-flight at end of this note

- Full HE n=164 ZAR eval (GPU 0)
- Full GSM8K n=1319 ZAR eval (GPU 1)

## Result — ZAR HE n=164 (2026-05-22 12:32Z)

cleaned pass@1 = **0.4207**, wall **1104s** (6.73s/sample), NFE 30699 (~187/sample).

Comparison:
| Method | cleaned pass@1 | wall (s) | s/sample |
|---|---|---|---|
| v3 K=384 | 0.4756 | 2361 | 14.40 |
| Fast-dLLM | 0.4329 | 1278 | 7.79 |
| **ZAR (w=8,α=8,β=32)** | **0.4207** | **1104** | **6.73** |
| v4 window-only | 0.3537 | 1669 | 10.18 |

**Hypothesis verdict**: pre-registered criterion "ZAR HE > 0.40 → strong evidence
prompt-tail refresh closes v4 gap" PASSES (0.4207 > 0.40, +6.7pp over v4).
ZAR doesn't beat Fast-dLLM on accuracy (−1.2pp) but is **fastest on HE
among all four methods** (1.16× over Fast-dLLM).

ZAR is faster than v4 despite refreshing ~7× more positions per step
because (a) extended warmup (8 vs 2) initializes K/V more accurately, (b)
prompt-tail refresh + lookahead push more positions over θ=0.9 confidence
per step, so blocks terminate earlier (fewer total cheap steps).

Open question: ZAR < Fast-dLLM on accuracy by 1.2pp. Candidates:
1. β=32 too small — Fast-dLLM refreshes full prompt at boundaries
2. α=8 too small — next block insufficiently warmed
3. drift in d128_512 also contributes (was 0.07 by step 50 in our measurement)

Next step (after GSM8K returns): ablate (β, α, warmup) — particularly
β=64, β=128, α=16, warmup=16.

## Decision criteria

Pre-registered:
- ZAR HE pass@1 > 0.40 cleaned: **strong evidence** that prompt-tail
  refresh closes the v4 gap on code tasks
- ZAR HE pass@1 ≈ 0.35-0.40: partial; prompt-tail alone not enough,
  need to reconsider α or include drift-driven refresh
- ZAR HE pass@1 ≤ 0.36: **refuted** — prompt-tail refresh doesn't help on
  HE; the v4-vs-v3 gap is NOT primarily about prompt staleness
- ZAR GSM8K strict ≥ 0.50: maintains v4's math win
- ZAR GSM8K strict ≤ 0.49: prompt-tail refresh hurts math (interesting
  asymmetry that needs explanation)

## Related

- [[session_state_2026_05_22]] — v4 window-only + per-layer schedule
  experiments; the trigger for this analysis.
- [[phase14a_e2e_results]] — v3 baseline numbers.
