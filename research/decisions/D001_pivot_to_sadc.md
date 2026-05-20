---
title: Pivot from Path A (rank-r predictor / cascade) to SADC (Spectral Active Deep Cache)
date: 2026-05-13
status: chosen
---

# Why we pivoted

## What Path A tried

Path A reframed the deep-layer hidden update as **analytically computable**:
- Attention: Sherman-Morrison rank-2 update (closed form when only K[cat_b], V[cat_b] change).
- FFN: Jacobian linearization (first-order Taylor).
Plus an offline rank-r predictor that learns a fixed basis for ΔH.

## What we learned (phases 9–10, day 1–5)

1. **Per-layer Path A is locally correct.** Day 3 cos(Δ_analytic, Δ_true) ≈ 0.86 for a single deep layer with baseline upstream.
2. **Cascade is catastrophic.** Day 4 17-layer cascade → final logits cos 0.97 / argmax 70%. Day-5 spectral analysis showed the cause: **per-deep-layer Jacobian amplification 1.18–1.43, average ≈ 1.27**, and per-layer rotation cos(J·Δ_in, Δ_in) ≈ 0.66–0.79. Compounded over 17 layers, small per-layer error blows up.
3. **Phase 10 production decoding: 0/2.** Garbage output, confirming cascade is lossy for actual decoding.
4. **Rank-r predictor (phase 4, 4b) failed for the same reason**: basis rotates per step, learned global basis can't chase.

## The new framing — SADC

> Deep dLLM caching is not a question of whether deep states are compressible,
> but whether their **error growth is controlled**.

Each (layer l, step t) partitions positions into three groups:
- **A_l^t** — active: full recompute (newly unmasked + window + sink + frontier + adjacent)
- **C_l^t** — corrected: cached + low-rank tangent correction (basis from current active deltas, NOT a learned global basis)
- **R_l^t** — reused: cached as-is

When-refresh comes from spectral profile:
$$K_l = \\lceil \\log(\\tau_l/e_0) / \\log A^{(l)} \\rceil$$
or threshold $e_t^{(l)} > \\tau_l$ where $e_{t+1}^{(l)} \\le A^{(l)} e_t^{(l)} + \\epsilon_t^{(l)}$.

## What we keep / drop

**Keep**:
- All capture infrastructure (`FullCapture`, `phase9_day3_composed.py`)
- Sherman-Morrison closed form (useful as *short-horizon* correction)
- Spectral analysis (day 5 — becomes the formal when-refresh result)
- Per-layer ΔH low-rank observation (motivates corrected set, not replacement)

**Drop**:
- Path A as *production decoding* (phase 10 form)
- Offline-learned rank-r ΔH predictor (phase 4, 4b)

## Differentiation from prior work

| Method | Decision unit | Decision signal |
|---|---|---|
| Fast-dLLM DualCache | block-wise | block boundary |
| Elastic-Cache | layer | attention drift threshold |
| dLLM-Cache | response feature | feature similarity |
| d²Cache | token | fine-grained selection |
| DyLLM / ES-dLLM | layer + position | stable-token sparsity |
| **SADC (ours)** | (layer × position) | **spectral error budget + observed active deltas** |

## Immediate next step

`phase11_oracle_active_set.py` — measure **oracle ceiling**: per (layer l, top-m by ||ΔH||_2), what's the cos_logit / argmax_agreement vs baseline? This Figure determines whether SADC is viable. If small m (say ≤ 32) is sufficient for deep layers → method survives.

See [[H001_active_set_compressibility]] for the live hypothesis.
