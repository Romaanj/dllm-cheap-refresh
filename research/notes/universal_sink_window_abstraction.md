# Universal sink+window abstraction for MDM heads (cross-LLaDA-Dream)

Date: 2026-05-17
Related: [[H003_mask_binder_computation]]

## Core claim

> Every attention head in MDM diffusion LMs is well-approximated as:
>   **head_attention(q) = (sink_set) ∪ (local_window(q, W))**
>
> The **sink set identity** and **window size** are model-specific (LLaDA's
> sinks are punct + mask, Dream's sinks are fixed positional). The
> **sink+window dichotomy** itself is universal.

This abstraction is the basis for cross-model sparse attention design.

## Evidence — published + our data

| Property | LLaDA-8B | Dream-7B | Source |
|---|---|---|---|
| Heads have sinks | yes (punct, mask) | yes (pos 256) | 2510.15731, our probe |
| Heads have local window component | yes (bidir_asym_L 40%, mask_binder ±5) | yes (bidir_asym_L 20%) | SparseD figs 7-8, our probe |
| Patterns stable across denoising steps | confirmed | confirmed | SparseD obs 2 |
| Patterns vary per head | confirmed | confirmed | SparseD obs 1, our cluster fractions |
| Left bias (79% of mass) | 79.6% | 79.0% | our probe |

## What is universal vs model-specific

**Universal (across LLaDA + Dream)**:
- Each head is decomposable into a small **sink set** + a **local window**.
- Patterns stable across denoising steps → static at inference.
- Left bias 79% in both models → directional inductive bias is a model-class
  property, not single-model accident.
- ~20% of heads do wide-left-context (bidir_asym_L exists in both).

**Model-specific**:
- *Sink identity*:
  - LLaDA punct_sink (35%): global dot/newline tokens.
  - LLaDA mask_binder (20%): mask tokens in local ±5 window.
  - Dream stationary_sink (62%): single position (256 in 5-shot setup).
- *Window size for the wide-context cluster*:
  - LLaDA bidir_asym_L window ≈ 150 positions (W80=18 in 32-block measure).
  - Dream bidir_asym_L window — not yet measured separately.

## How this differs from published work

| Paper | Coverage | What they did | Missing |
|---|---|---|---|
| 2510.15731 (Attention Sinks in DLMs) | Both | Token-identity of sinks per model | No window analysis, no head clustering, no fractions |
| 2510.09309 (MaskKV) | Both | Continuous Information/Structure score | No discrete taxonomy, no cross-model alignment |
| 2509.24014 (SparseD) | Both | Per-head dynamic sparse attention | No named clusters, no fractions, no cross-model |
| 2510.14973 (Elastic-Cache) | LLaDA | Aggregate "mask is length-bias" | Wrong about LLaDA (contradicted by 20% mb cluster) |

**Our novelty in this abstraction**:
1. *Named* taxonomy ({sink-only, window-only, mixed}) testable per-head.
2. *Counted* fractions per model.
3. *Cross-model alignment* statement: same abstraction, different parameters.
4. *Design principle*: probe-discover (sink set, window W) per head, then
   apply sparse attention with those exact parameters.

## Method implication

```
Method (model-agnostic):
  1. Probe model M with a small calibration set (3-30 prompts).
  2. For each (layer, head), measure:
     - sink_set(M, layer, head) = positions getting top concentrated mass
     - window_W(M, layer, head)  = half-width capturing 80% of remaining mass
  3. At inference, replace dense attention with:
     attn(q) = softmax(Q_q · K_{sink_set ∪ [q-W, q+W]} / sqrt(d_h)) · V_{...}
  4. Compute saved = 1 - (|sink_set| + 2W) / L on average.
```

LLaDA estimated savings: ~83% compute reduction (mask_binder ±5, punct sinks
small).
Dream estimated savings: ~95%+ compute reduction (1 sink position + window).

## Open questions (must verify before paper claim)

1. Does the **sink+window** approximation hold tightly per head? Need to
   measure: for each head, % of attention mass captured by (its sink set ∪
   its window).
2. Does this approximation preserve generation quality? Need to: replace
   dense attention with sink+window for a subset of heads, eval accuracy.
3. Does sink_set + window transfer across tasks (we have IoU 73-79% for
   LLaDA cluster identity, but not yet for sink set IDs)?
4. MMaDA — third dLLM as sanity check.

## Status

Abstraction proposed, supported by published observations + our data.
**Not yet empirically validated as sparse attention design**.
Next concrete experiment: per-head (sink_set ∪ window) attention mass
coverage analysis on existing probe data.
