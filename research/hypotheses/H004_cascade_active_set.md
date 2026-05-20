---
title: H004 — Cascade-respecting per-(layer × position) active set
date: 2026-05-18
status: supported
related: [[H001_active_set_compressibility]], [[R26_scope_incremental_negative_result]], [[q1a_quest_dllm_oracle]], [[D001_pivot_to_sadc]], [[phase14a_e2e_results]]
---

# Problem (user's core question)

Elastic-Cache and other dLLM caching methods refresh entire layers when
triggered (whole sequence at chosen layer). User intuition:

> If at each step only a small subset of positions has materially changed
> hidden state, can we refresh **only that subset at every layer** (both
> shallow AND deep) and still recover baseline logits?

Shallow layers do "information scanning" (dense, heterogeneous heads); deep
layers do "actual mixing" (sparser, content-concentrated). If both admit a
small per-layer active set, total compute drops as
  cost ≈ Σ_l |A_l| × (attn + ffn cost per position)
instead of N × L × cost_per_position.

H001 showed *single-layer* oracle is viable (m=64 at L25 → cos 0.997). H004
asks the **cascade** version: when applied at every layer simultaneously,
does the per-layer active set stay bounded, or does it cascade outward
toward |A_l| = L?

# Claim

For LLaDA-8B at L≈1316, gen=256, block=32, there exists an **oracle**
per-layer active set policy A_l = top-K_l positions by ||H_l^t − H_l^{t−1}||
such that:

$$
\cos(\text{logits}_{\text{hybrid}(K_1,...,K_N)},\ \text{logits}_{\text{baseline}}) \ge 0.99
$$

with **mean K_l / L ≤ 0.4** (averaged over layers and steps).

Hybrid forward (oracle):
```
hybrid_in[0] = cur_input[0]                         # current embedding
for l = 0..N-1:
  out_l = layer_l(hybrid_in[l])                     # full attention
  hybrid_in[l+1] = out_l                            # refresh A_l
  hybrid_in[l+1][~A_l] = prev_block_out[l][~A_l]    # reuse prev for rest
logits = lm_head(ln_f(hybrid_in[N]))
```

# Sub-hypotheses

- **H004a (uniform-K viability)**: A uniform per-layer budget K_uniform ≤
  0.4 × L (≈ 526 at L=1316) achieves mean cos_logits ≥ 0.99 averaged over
  block-mask positions across step pairs.

- **H004b (layer-stratified savings)**: Per-layer "true active set" size
  |{p : ||ΔH_l[p]||_2 > τ}| grows with layer but plateaus — shallow L<8
  is wide (≥ 60%), mid L=8-15 narrows, deep L≥16 stays narrow (≤ 30%) for
  τ tuned to capture 95% of ||ΔH_l||_F^2 mass.

- **H004c (cascade containment)**: At fixed K_l, cos_logits remains
  monotone in K AND degrades smoothly (no cliff). Refuted if there is a
  threshold K* below which cos suddenly collapses (indicating cascade
  divergence is unbounded below the threshold).

- **H004d (compute saving estimate)**: At K_uniform such that cos ≥ 0.99,
  effective compute ratio Σ_l K_l / (N × L) ≤ 0.40, i.e. ≥ 2.5× attention
  speedup (oracle ceiling).

# Falsification

H004 is refuted if **any** of:
- H004a: no uniform K achieves cos ≥ 0.99 at ≤ 0.4 × L (would require
  >0.5 × L or fail to converge → no compute savings).
- H004b: deep-layer true-active count is ≥ 50% of L on average (no
  layer-stratified opportunity, contradicts Q1a finding).
- H004c: cos drops abruptly at some K* (cascade unbounded → no clean
  schedule).
- H004d: Σ_l K_l / (N × L) > 0.6 at the operating point → savings too
  modest to motivate a method.

# Method (Phase 12)

Script: `phase12_cascade_oracle.py`.

Capture block_in & block_out at **all 32 layers** for two consecutive
decoding steps (prev = t−1, cur = t). For each (step pair, K), run the
hybrid cascade forward defined above with A_l = top-K positions by
||cur_block_out[l] − prev_block_out[l]||_2.

Record per (sample, step, K):
- `cos_logits_block_masks` at current-block mask positions
- `argmax_agreement`
- `top5_agreement`
- per-layer |A_l| effective (= K)
- per-layer true-active counts at thresholds {1e-2, 5e-2, 1e-1, 2e-1} of
  per-position delta-norm vs per-layer mean delta-norm
- per-layer ||ΔH_l||_F coverage by top-K positions (concentration check)

Settings:
- model: GSAI-ML/LLaDA-8B-Instruct
- gen-length: 256, block: 32, steps: 256, dual-cache OFF
- num_samples: 8 (split 4 per GPU)
- num_fewshot: 5 (GSM8K test)
- probe stride: 16 step pairs per sample
- K grid: {0, 16, 32, 64, 128, 256, 512, T_full}

Expected wall time: ~15-20 min per GPU.

# Pre-registered decision rule

After full sweep:
- If H004a + H004d both PASS (K_uniform ≤ 0.4 L gives cos ≥ 0.99) →
  proceed to Phase 13 (cheap online estimator of A_l).
- If H004b PASSES but H004a FAILS → consider per-layer adaptive K (try
  K_l proportional to delta-norm concentration).
- If H004c FAILS (cliff exists) → cascade is structurally unbounded under
  reuse-paradigm; pivot back to characterization paper.

# Status

- 2026-05-18: Registered. Phase 12 script under implementation.
- 2026-05-18: Full sweep run (8 samples × 15 probes × 10 K = 1200 rows).
  **STRONGLY SUPPORTED**. H004a/c/d PASS. H004b PASS on magnitude
  (deep ≤ 10% true-active) but direction inverted (deep has MORE active
  than shallow, not fewer). Cascade containment confirmed: smooth concave
  curve, K=128 (10% L) → cos 0.998; K=384 (28% L) → argmax 0.97.
  See [[phase12_cascade_oracle]] for full numbers and method implications.
- 2026-05-18: Phase 13a + 13b. Cheap (lag-1 history) estimator captures
  80%+ of oracle gain at K≥256. E3 (U_t + window4 + lag-1) reaches
  argmax 0.952 at K=384 (oracle 0.966, gap 1.4pp). **Method route viable.**
  See [[phase13a_oracle_geometry]] + [[phase13b_cheap_estimator]].
- 2026-05-18: **Phase 14a end-to-end VERIFIED**. GSM8K n=100: baseline
  0.810 vs cheap_e3 0.820 (gap +0.01, SE 0.04). HumanEval n=164: baseline
  0.372 vs cheap_e3 0.384 (gap +0.012, SE 0.04). Both within statistical
  parity. Method does NOT catastrophically degrade over 254 consecutive
  cheap forwards. **H004 strongly supported across all axes including
  end-to-end accuracy. Status: SUPPORTED → close hypothesis.** Next:
  Phase 14b masked kernel for real wall-clock speedup, SPA-Cache head-to-head.
  See [[phase14a_e2e_results]].
