---
name: method-evolution-directions-2026-05-20
description: Candidate directions for advancing the E3 cheap-refresh method (lag-1 + U_t + window-4 + sparse-Q + KV cache). Surveyed from prior phase findings and lit-review notes. Use this as a starting menu when the new server picks up method-development work.
metadata:
  type: project
---

# Method evolution directions — entry point for new-server method development

The benchmark re-runs (post-D002, chat ON + gen=512) are being executed on the source server and will be synced when complete (see [[session-state-2026-05-20]]). This note is for the *parallel* track: methodology improvement on the new server.

## What we currently have (E3 baseline)

- **Refresh signal**: lag-1 (per-position confidence change) + remaining-unmask budget $U_t$ + window-4 (local neighbors of unmask events)
- **Active set $A_l$**: built from these signals, budget controlled by $K$ (stepped on LLaDA, uniform on Dream)
- **Forward path**: sparse-Q forward (only $A_l$ positions get fresh Q/K/V), rest reuse cached
- **Composition**: warm-up 2 baseline forwards, then E3 + KV cache, $\theta{=}0.9$ confidence-threshold parallel commit
- **Known numbers** (gen=256, chat OFF — see D002 for protocol caveat):
  - LLaDA GSM8K: 0.7612 flex / 0.4943 strict @ 4.48× wall
  - LLaDA HumanEval: 0.427 pass@1 @ 2.46×
  - Dream GSM8K: 0.7309 flex / 0.7066 strict @ 7.62×
  - Dream HumanEval: 0.524 pass@1 @ 2.63×

## Where we are weak (motivation)

1. **Dream side has unexplained $-10\text{pp}$ accuracy drop** on hard tasks despite oracle ceiling being equally high (see [[phase12_cascade_oracle_dream]]). Diagnosis: bottleneck is *cheap estimator quality* or *KV-cache mechanism*, not refresh budget.
2. **Cheap-vs-oracle penalty is $1.5{\times}$ budget** ([[phase13b_cheap_estimator]]). Closing this gap = nearly free speedup.
3. **Fixed $K$ schedule** — adaptive $K_t$ likely Pareto-better, especially in early steps where refresh need is highest.
4. **Elastic-Cache matches/beats us on HumanEval accuracy** with a different signal (attention-drift cos sim, per layer). Worth understanding what they capture that we miss.

## Direction menu

### D-A. Improve the lag-1 signal itself
- **Lag-2 or higher** — does keeping a 2-step or 3-step delta capture more A_oracle than lag-1? [[phase13b_cheap_estimator]] only tested lag-1.
- **Multi-signal fusion** — current E3 fuses 3 signals (lag, $U_t$, window). Add attention-drift (Elastic-Cache style) as a 4th channel and learn weights?
- **Per-head vs per-position** — current refresh is at position granularity. [[fastgen_head_pattern_probe]] + [[universal_sink_window_abstraction]] suggest per-head heterogeneity matters. Per-head $A_l$?
- **Confidence vs entropy as the cheap signal** — lag-1 uses argmax confidence. Top-2 margin or full entropy might detect drift differently.

### D-B. Adaptive budget $K_t$
- Current: stepped $K$ on LLaDA, uniform on Dream. Why those work asymmetrically? ([[phase12_cascade_oracle_dream]] suggests Dream's deeper concentration favors uniform).
- **Online $K_t$** driven by $\theta$-threshold yield: if many positions clear $\theta$ at step $t$, the remaining masks are easier and $K_{t+1}$ can shrink.
- **Per-layer $K^\ell$** — early layers may need much smaller $K$ than deep (cascade-respecting per [[phase12_cascade_oracle]]).

### D-C. Forward kernel
- Current sparse-Q forward computes Q only for $A_l$ positions but uses full K/V. Could we **sparse-K** for unmask events too?
- The "planned masked kernel" mentioned in [[novelty_positioning_2026_05_18]] is still unwritten — would consolidate sparse forward as a real custom kernel rather than tensor reshuffling.

### D-D. Trigger / refresh policy
- Current: refresh once per step (every forward pass). Elastic-Cache shows per-layer trigger gives $13\%$ cache update frequency at no accuracy cost.
- **Hybrid trigger**: skip whole forward passes (block-level), only refresh on drift.
- **Hierarchical refresh**: refresh "core" $A_l$ every step, "periphery" $A_l$ every $k$ steps.

### D-E. Composition with parallel
- Current: $\theta{=}0.9$ greedy commit. Could explore **gradient-aware composition** (commit more cautiously when E3 estimator entropy is high).
- **Repair**: if a committed token's neighborhood lag-1 spikes immediately after, allow re-mask (currently no remask).

### D-F. New tasks / robustness
- **MATH-500** — stress reasoning chain; Dynamic-DLLM's APD failure mode would amplify, our cheap refresh likely OK.
- **MBPP** — second code benchmark; Elastic-Cache vs Ours on a different code distribution.
- **Long-form** (BBH / IFEval) — exposes window-method (Elastic-Cache) assumption; positions us favorably.

### D-G. Apply method to other dLLMs
- DKR-Diff, Edit-Diff, SDD families exist (see [[lit/...]] notes). Our cheap E3 should transfer if those use confidence-style sampling.

## Suggested first move on new server

Pick ONE of:
1. **D-B online $K_t$** — quick to try, validates "adaptive helps" hypothesis, low risk.
2. **D-A multi-signal fusion** with attention-drift channel — tests against Elastic-Cache's signal directly.
3. **D-D hybrid trigger** — biggest potential speedup gain; medium risk.

For each, the experiment template:
1. Implement in `methods/ours/cheap_e3_inference_llada.py` (or new variant module)
2. Sanity check on small subset (N=10) of GSM8K, verify acc within 1pp of baseline
3. Full run on GSM8K $N=1319$
4. Compare to baseline E3 row in `tables.tex`
5. Write findings to `research/notes/<short_name>.md`, link from `INDEX.md`

## What NOT to do on new server (avoid duplicate work)

- **Do NOT re-run published-baseline HumanEval evals** — source server is doing all 7 (see Section 8 of [[session-state-2026-05-20]]). Just `rsync` the result samples when ready.
- **Do NOT touch `tables.tex`** for the *headline* numbers until source-server re-runs land — would conflict.
- Adding new rows for methodology variants is fine: append a new ablation table.

## Pointers

- Prior phase notes (in `research/notes/`): phase12_cascade_oracle, phase12_cascade_oracle_dream, phase13a_oracle_geometry, phase13b_cheap_estimator, phase14a_e2e_results, R26_scope_incremental_negative_result, universal_sink_window_abstraction, fastgen_head_pattern_probe
- Hypotheses: H001-H004 — open questions worth re-revisiting
- Decision D001 (SADC pivot), D002 (HumanEval protocol)
- Lit notes: spa_cache, dinfer, dkv_cache, sparsed, maskkv, retroattention, window_diffusion, elastic_cache, fast_dllm, dynamic_dllm — closest prior art
