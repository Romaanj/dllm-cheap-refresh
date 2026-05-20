---
title: Phase 12 — Cascade-respecting per-(layer × position) refresh oracle
date: 2026-05-18
status: complete
related: [[H004_cascade_active_set]], [[H001_active_set_compressibility]], [[R26_scope_incremental_negative_result]]
---

# TL;DR

**User's core question — "can we refresh only a small subset of positions at
every layer (shallow + deep) and still recover baseline logits?" — gets a
strong YES under oracle.**

- At `K_uniform = 128` (10% of L=1316), mean cos_logits = **0.998** at
  block-mask positions.
- At `K_uniform = 384` (28% of L), mean argmax agreement = **0.97**.
- Curve is **smooth and monotone** — no cliff. Cascade is containable.
- Plot: `results_phase12_cascade_oracle/k_curve.png`,
  `results_phase12_cascade_oracle/per_layer.png`.

This is the **positive complement** to [[R26_scope_incremental_negative_result]]:
**reuse paradigm fails**, but **refresh-only paradigm (always recompute
A_l forward, never trust stale output for A_l positions) works** because
the cascade absorbs staleness into A_l rather than amplifying it.

# Setup

- Script: `phase12_cascade_oracle.py`
- Model: GSAI-ML/LLaDA-8B-Instruct
- 8 GSM8K test samples, 5-shot, gen=256, block=32, steps=256, **DualCache OFF**
- Probe stride: 16 → 15 probe step pairs per sample → 120 total
- K grid: {0, 32, 64, 128, 256, 384, 512, 768, 1024, full(=T)} uniform
- All 32 layers' block_in/block_out captured per step
- Hybrid forward at each (step, K):
  - A_l = top-K positions by ||cur_block_out[l] − prev_block_out[l]||_2
  - layer l output: out_l[A_l] (refreshed) ⊕ prev_block_out[l][~A_l] (reused)
  - Compute final hidden + lm_head

# Results

## K-curve (mean across 120 step pairs)

| K | K/T | cos_mean | cos_p10 | argmax_mean | argmax_p10 | top5_mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.9853 | 0.9548 | 0.865 | 0.562 | 0.982 |
| 32 | 0.024 | 0.9911 | 0.9748 | 0.883 | 0.688 | 0.992 |
| 64 | 0.047 | 0.9955 | 0.9894 | 0.899 | 0.688 | 0.994 |
| 128 | 0.095 | **0.9980** | 0.9959 | 0.929 | 0.812 | 0.998 |
| 256 | 0.189 | 0.9991 | 0.9983 | 0.945 | 0.841 | 1.000 |
| 384 | 0.284 | 0.9995 | 0.9990 | **0.966** | 0.906 | 1.000 |
| 512 | 0.379 | 0.9997 | 0.9993 | 0.969 | 0.906 | 1.000 |
| 768 | 0.568 | 0.9998 | 0.9997 | 0.974 | 0.934 | 1.000 |
| 1024 | 0.758 | 0.9999 | 0.9997 | 0.982 | 0.938 | 1.000 |
| full | 1.000 | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |

Observations:
- **Cascade containment is real**. K=0 (pure prev reuse) already gives
  cos 0.985 and argmax 0.865 — the cascade hybrid forward IS forgiving.
- Concave curve: most argmax gain happens between K=0 and K=384. Beyond
  K=512, gains are minimal (0.97 → 0.98 over 30%-pt budget increase).
- `top5_mean ≈ 1.0 at K=256+` — the true token always lands in top-5.

## Per-layer "true active set" — positions with |ΔH_l[p]| > c × layer mean

Counterintuitive finding: **deep layers have MORE high-delta positions, not
fewer** (against [[q1a_quest_dllm_oracle]]'s attention-sparsity finding).

| | shallow L<8 | deep L≥16 | ratio |
|---|---:|---:|---:|
| count > 1× layer mean | 270 (20% L) | 370 (28% L) | 1.37× |
| count > 2× layer mean | 64  (4.9% L) | 126 (9.6% L) | 1.95× |
| count > 4× layer mean | 17  (1.3% L) | 36  (2.7% L) | 2.12× |

Mechanism: cross-layer cascade r=0.86 means deltas accumulate as we descend.
By layer 31, every layer's small movements have nudged many positions —
even though attention itself is sparser deep ([[q1a_quest_dllm_oracle]]).

→ **Per-layer concentration ≠ attention concentration**. Two different
phenomena. Q1a measures *which keys queries attend to* (sparse deep).
H004 measures *which positions' hidden state changes step-to-step* (more
diffuse deep due to cascade).

## Per-layer top-K coverage of ||ΔH_l||

Shallow has higher concentration at small K, comparable at large K:

| layer | K=32 | K=128 | K=256 | K=512 | K=1024 |
|---:|---:|---:|---:|---:|---:|
| 0  | 0.345 | 0.493 | 0.593 | 0.720 | 0.910 |
| 8  | 0.225 | 0.362 | 0.482 | 0.654 | 0.896 |
| 16 | 0.172 | 0.328 | 0.454 | 0.633 | 0.883 |
| 24 | 0.157 | 0.352 | 0.508 | 0.703 | 0.913 |
| 31 | 0.176 | 0.398 | 0.561 | 0.748 | 0.930 |

# H004 verdict (pre-registered)

- **H004a (K ≤ 0.4L → cos ≥ 0.99)**: **PASS** — even K=32 (2.4% of L)
  gives cos 0.991. K=128 (10%) gives 0.998.
- **H004b (deep ≤ 30% true-active)**: **PASS on magnitude**, but
  **direction is inverted** — deep has more active positions than shallow
  (1.95× ratio for >2×mean), not fewer. The 30% threshold still holds
  because deep maxes at ~10%.
- **H004c (monotone, no cliff)**: **PASS** — smooth concave curve, no
  threshold below which cascade explodes.
- **H004d (compute saving)**: **PASS** on loose criterion (cos≥0.99
  at K=32 → 40× attention speedup oracle ceiling); **strict criterion**
  argmax≥0.95 needs K=384 (28% L) → 3.5× speedup ceiling.

# What this rules in / out

**Rules IN (next path)**:
- "Refresh-only" framing — the runtime ALWAYS computes the active set's
  layer output (no stale reuse for refreshed positions); stale state is
  only used as substitute for positions outside A_l. This avoids R26's
  failure (which trusted stale outputs at A_l positions through aliased
  reads).
- Per-layer K can be uniform; further gain from per-layer adaptive K
  (smaller shallow, bigger deep) is plausible given the per-layer
  concentration profile but not yet measured.

**Rules OUT**:
- Strict 100% argmax preservation needs full refresh. There's a small
  residual error (1-3% argmax positions) that no K < full clears.
  Acceptable for a 3-10× speedup target but not for "lossless decode".

# Implications for method design

Three questions remain before this becomes a viable method:

1. **Cheap online estimator for A_l**: oracle uses
   ||cur_block_out[l] − prev_block_out[l]|| which requires computing
   cur_block_out[l] first — defeats the purpose. Candidate proxies:
   - newly-unmasked positions U_t (always include)
   - prev-step's per-position delta norm (history)
   - neighbors of recently-unmasked (mask_binder window-aware)
   - cluster identity (per-head dispatch from [[fastgen_head_pattern_probe]])
   - low-cost first-pass approximation of upstream delta
2. **End-to-end wall-clock test**: oracle counts query positions, but
   actual cost depends on attention kernel + KV refresh cost. Need to
   either (a) implement masked attention that skips ~A_l rows, or
   (b) measure FLOPS-equivalent of K/T ratio.
3. **Compositionality with existing methods**: Fast-dLLM block-decode +
   Phase 12 per-position refresh + Q1a per-head KV sparsity — do they
   multiply or interfere?

# Next decision

Three candidate paths:

| path | what | risk |
|---|---|---|
| **A (recommended)** | Phase 13: design + benchmark cheap A_l estimator. Start with U_t + history + recency-window. Target argmax≥0.95 (K=384) operating point. | Estimator quality may collapse the oracle ceiling |
| B | Phase 13b: per-layer adaptive K based on concentration profile (smaller shallow, bigger deep). Re-run oracle. | Marginal gain over uniform K, distraction |
| C | Skip estimator; write characterization paper. Phase 12 + R26 + Q1a + H003 form a complete observational story. | Loses the method contribution |

User intuition validation: **shallow=scan, deep=mix** — partially correct.
Deep layers ARE where most actively changing positions live, just because
cascade accumulates. The scan-then-mix narrative still tracks but the
"savings live in deep sparsity" intuition flips: savings live in BOTH
because both have low |true-active|/L (deep 10%, shallow 5%).

# Files

- `phase12_cascade_oracle.py` — measurement script
- `analyze_phase12_cascade.py` — verdict computation
- `plot_phase12_cascade.py` — k_curve + per_layer plots
- `results_phase12_cascade_oracle/gpu{0,1}/cascade_oracle.jsonl` — 1200 rows total
- `results_phase12_cascade_oracle/k_curve.png`
- `results_phase12_cascade_oracle/per_layer.png`
- `analysis_phase12_cascade.txt` — raw analysis dump
