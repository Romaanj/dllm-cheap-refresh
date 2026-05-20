---
title: H001 — Active-set compressibility of deep ΔH
date: 2026-05-13
status: open
---

# Claim

For every deep layer $l$ at every step $t$, there exists a small **active set** $A_l^t \\subseteq [0, T)$ with $|A_l^t| \\ll T$ such that:

$$
\\text{cos}\\Big(\\text{logits}_{\\text{baseline}},\\ \\text{logits}_{\\text{with cache}[l] + \\text{recompute on } A_l^t}\\Big) \\ge 0.99
$$

i.e. recomputing only the top-(m) positions by $\\|\\Delta H_l[p]\\|_2$ (cache reuse elsewhere at depth $l$) reproduces baseline logits.

# Sub-hypotheses

- **H001a** (deep-layer ceiling): $|A_l^t| \\le 32$ suffices for $l \\in [15, 31]$ on GSM8K gen=256.
- **H001b** (layer monotonicity): required $m_l$ grows with $l$ up to a point, then plateaus or shrinks (because cascade depth shrinks).
- **H001c** (delta concentration): top-m positions account for $\\ge 70\\%$ of $\\|\\Delta H_l\\|^2_F$.

# Method

`phase11_oracle_active_set.py` — for each step pair, each probe layer, each $m \\in \\{0, 1, 2, 4, 8, 16, 32, 64\\}$:
1. Build hybrid $H_l^t \\leftarrow H_l^{t-1}$ with top-$m$ positions replaced by $H_l^t$ (oracle: knows true $\\Delta H_l$).
2. Forward layers $l..31$ from hybrid.
3. Measure cos_logits, argmax_agreement, cos_hidden_final at block-mask positions.

# Outcomes that would refute

- H001a refuted if required $m$ for any deep $l$ exceeds 64 on average (would need too-large active set; SADC degenerates to full recompute).
- H001b refuted if curve is non-monotone or chaotic across layers (suggests no clean schedule).
- H001c refuted if delta_norm_coverage at top-32 < 50% on average (long tail of small but non-trivial perturbations dominates → no concentration to exploit).

# Status

- 2026-05-13: Smoke test (1 sample × gen=128 × layers={15,25} × m={0,4,64,128}) shows promising shape:
  - layer 15, m=4: cos=0.996, argmax=89%
  - layer 15, m=64: cos=0.9998, argmax=98%
  - layer 25, m=4: cos=0.987, argmax=76%
  - layer 25, m=64: cos=0.997, argmax=89%
  Pattern: small $m$ gives near-lossless for shallow probe-layers, larger $m$ needed deeper. Will confirm with full sweep (`b8nz3a5uf`).
