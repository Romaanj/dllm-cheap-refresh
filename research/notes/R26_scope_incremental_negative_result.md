# R26 — SCOPE-DRIVEN INCREMENTAL framework: negative result + diagnosis

Date: 2026-05-18
Related: [[H003_mask_binder_computation]], [[universal_sink_window_abstraction]]

## TL;DR

Cache-reuse methods (static scope-based AND dynamic drift-based) for dLLM
KV compression **fundamentally fail** in our experiments. None approach
R_curr's 0.755 baseline. We identify and quantify the mechanisms responsible:
bidir attention + cross-layer shift propagation + diffuse attention + peak
mobility.

## Goal

Design and validate SCOPE-DRIVEN INCREMENTAL attention framework:
  - Per-(layer, head) cache of attention outputs
  - Each step, identify newly-unmasked positions U_k
  - For each (h, q): if scope(h) ∩ U_k = ∅ → reuse cached output
  - Else: recompute

Target: ≥0.76 GSM8K 5-shot full 1319 (match baseline 0.78).

## Implementations tested

| Variant | Description | N=50 result | Status |
|---|---|---|---|
| V1 scope | cluster-based W: mb=10, ps=5, bd=150 | 0.48 | fail |
| V2 scope (wider) | cluster-based W: mb=32, ps=16, bd=300 | 0.54 | fail |
| V0 drift t=0.1 | attn cos-sim drift, single threshold | 0.25 (N=20) | fail |
| V0 drift t=0.05 | tighter threshold | 0.35 (N=20) | fail |
| (baseline R_curr) | dual cache + R_curr floor | 0.755 (full 1319) | reference |
| (baseline vanilla) | full KV no compression | 0.78 (full 1319) | reference |

All proposed variants fail by 20–55pp vs R_curr.

## Key observations (rich diagnosis)

### 1. Attention is moderately peaked per-step but peaks shift

```
Per-step concentration:
  top-1: 10% mass    top-16: 49% mass    top-64: 72% mass

Pooled across steps (loses shift info):
  top-1:  5%   top-16: 33%   top-64: 56%

Top-1 position shift between consecutive saved steps:
  mean fraction shifted: 57.4%
  median shift distance: 21 positions
  p25/p50/p75: 9 / 21 / 66
```

This **invalidates static scope** — the right positions to keep change every step.

### 2. Per-cluster shift dynamics

| cluster | shift rate | median dist |
|---|---|---|
| stationary_sink | 2.5% | (in prompt) |
| punct_sink | 40.1% | 38 |
| bidir_asym_L | 60.7% | 22 |
| mask_binder | **80.0%** | **11** (LOCAL) |
| frontier_tracker | 100% | 8 |

mask_binder (the cluster causally critical per knockout test) is the most
volatile (80% shift rate) but shifts locally (median 11 positions). Static
scope ±5 misses these shifts.

### 3. Cross-layer shift correlation (uniform propagation)

```
adjacent layers (L, L+1) Pearson r = 0.86
Δ=5 layers: r = 0.78
Δ=10:       r = 0.67
Δ=20:       r = 0.52
mean off-diagonal r: 0.665
```

When unmask occurs, ALL layers shift together. Cannot selectively skip
specific layers — bidir attention propagates immediately.

### 4. Within-layer drift heterogeneity (shallow vs deep)

```
ratio (within-layer drift std / drift mean):
  shallow L0-7:   1.0-1.8 (highly heterogeneous, head-level matters)
  deep L11+:      0.5-0.7 (uniform, layer-level OK)
```

Shallow layers' heads are very diverse — layer-aggregate refresh loses
info. Deep layers more uniform.

### 5. Per-cluster mean attention drift (cos similarity 1 - x)

```
stationary_sink: 0.004 (barely changes)
punct_sink:      0.127
bidir_asym_L:    0.237
mask_binder:     0.352
frontier_tracker:0.653
```

Mass spread across clusters means single drift threshold can't gate
appropriately for all heads.

### 6. Frontier coupling (where peaks live wrt newly-unmasked)

| cluster | median dist to U_k | % shifts toward U_k |
|---|---|---|
| frontier_tracker | 1 | 86% |
| mask_binder | 9 | 54% |
| bidir_asym_L | 31 | 68% |
| punct_sink | 43 | 70% |

mask_binder lives near frontier but doesn't deterministically follow
it — random walk in window.

## Why cache reuse fails (mechanistic explanation)

### Failure mode 1: Cascading

```
Layer L step k: some attention outputs change → layer L+1 input changes
Layer L+1: even unaffected queries' Q differs → attention output differs
... cascades through 32 layers ...
```

V1 scope reuses outputs assuming scope is tight. Even with 80% mass
captured, 20% loss per layer compounds: 0.80^32 ≈ 0.001 cumulative.

Wider W (V2) reduces per-layer loss but compounds remain. 32 layers ×
small drift = significant divergence.

### Failure mode 2: Attention drift ≠ output drift

V0 used attention cos-sim drift. But:
- Same attention vector + different V values → different output
- V values at attended positions change when their hidden states change
  (via cascading)
- Drift signal underestimates output divergence

V0 reuses outputs when attention cos-sim is high, but output may have
drifted via V changes. Stale outputs propagate.

### Failure mode 3: dLLM bidir vs AR causal

AR cache compression (StreamingLLM, H2O, FastGen) works because past
is IMMUTABLE: once generated, past tokens' hidden states are frozen.
Only future affects past via causal mask = never.

dLLM is bidirectional: every unmask event affects ALL position's hidden
states at all layers (per our cross-layer correlation analysis). No
position is "frozen" — past tokens' representations evolve as masks
unmask elsewhere.

→ Cache reuse strategies that worked for AR cannot transfer.

## What does work (best accessible baseline)

R_curr (existing dual cache with mask floor extension): 0.755 full 1319.

R_curr's success comes from:
1. Block-warm-pass: re-compute everything at block boundaries → catches
   the strongest shifts (between blocks).
2. Dynamic per-head top-K based on attention score at warm pass → catches
   prompt-specific peak positions.
3. Static structural floor (punct + R_curr mask) → preserves anchors.

Even R_curr has a 2.5pp gap to baseline. K_ratio sweep (K=0.30, 0.50, 0.99)
confirms framework saturates near 0.76 — not a budget issue.

## Implications & framings

### Negative finding (workshop / short paper)

> "We empirically demonstrate that dLLM KV cache compression via cache reuse
> (whether static scope-based or dynamic drift-based) is fundamentally
> limited by bidirectional attention propagation. Through extensive
> characterization — peak mobility (57% shift rate), cross-layer
> propagation (r=0.86), and diffuse attention — we identify the
> mechanisms responsible. Methods that work for AR LLMs do not transfer."

This is a valid contribution: charts the impossibility boundary of an
approach, with mechanistic explanation.

### Positive observational findings (still publishable)

- **mask_binder cluster (~20% LLaDA heads)** — task/shot-invariant,
  causally critical (knockout → 0%), proximity-based (W80=5). LLaDA
  emergent property not in Dream-7B.
- **Cross-model divergence** — LLaDA's mask_binder is from-scratch MDM
  artifact; Dream's position-256 sink is AR-init artifact.
- **Cross-layer shift correlation** — empirically validates
  Elastic-Cache's layer-wise refresh design.
- **Frontier coupling per cluster** — different clusters live at different
  distances from decoding edge.

### Path β suggestions for future work

What MIGHT work that we didn't fully test:
1. **Layer-stratified hybrid**: full attention for deep layers (where
   cascading damages most), sparse for shallow (where heterogeneity allows
   per-head decisions).
2. **Refresh-based (no reuse)**: like Elastic-Cache but with head-level
   trigger. Refresh K layers downstream when triggered.
3. **Block-warm-pass enhancement**: extend R_curr's success — keep block
   warm pass but improve the per-head top-K signal (e.g., predict future
   shifts).
4. **Training-time intervention**: align LLaDA's emergent specialization
   with a known-compressible structure (long-term, not for current goal).

## Time spent

~5 hours GPU time across:
- E1 implementation + sanity (1h)
- E2 N=20 V1 (~20min)
- E3 N=50 V1 + V2 + V0 t=0.1 + t=0.05 (~2h)
- Diagnostic analyses (probes + analyses) (~1h)
- Misc (~1h)

Within 6h budget.

## Status

**STOPPED**: framework not viable at allotted budget. Negative finding
captured. Observational findings strengthened.

Next session candidates:
- Pivot to refresh-based (no-reuse) framework
- Or accept negative result + write up + paper as workshop
- Or layer-stratified hybrid as final attempt
