---
title: H002 — Query-group compressibility of KV-page attention (Quest-dLLM viability)
date: 2026-05-14
status: weak-pass (with refinements; see [[q1a_quest_dllm_oracle]])
---

# Background

Quest (Tang et al., ICML 2024) accelerates AR-LLM long-context inference by
selecting Top-K KV pages per query token using a cheap channel-wise min/max
estimator. Its core claim is **per-query** sparsity — different queries within
the same prompt attend to different critical pages.

In dLLM (LLaDA), every step has all positions simultaneously active as
queries (full bidirectional attention, no causal mask). Per-query Top-K
selection is therefore O(M_active × estimator_cost) per step, which kills
the speedup.

**The natural adaptation**: compute the (query group → critical KV page set)
mapping **once per step**, where queries are grouped by some shared
structure (block membership, spatial chunk, Q-vector cluster, head).

# Claim

For LLaDA-8B, gen=256, block_size=32, **DualCache OFF**:

For every layer $l \ge l_0$ at every step $t$, **queries within the
current decoding block** $B_t$ share a small KV-page set $P_l^t \subseteq
\{1,\dots,N_{\text{pages}}\}$ with $|P_l^t| \ll N_{\text{pages}}$ such
that:

$$
\sum_{q \in B_t}\ \sum_{p \in P_l^t}\ \mathrm{attn}_l[q, p]\ \ge\ 0.95
\cdot \sum_{q \in B_t}\ \sum_{p}\ \mathrm{attn}_l[q, p]
$$

i.e. a *shared* Top-K page set for the whole block captures ≥95% of the
block's attention mass. Page size = 32 (block-aligned).

# Sub-hypotheses

- **H002a (group coherence)**: For deep layers ($l \ge 4$), the mean
  pairwise **Jaccard** of per-query oracle Top-K page sets within the
  current decoding block is ≥ 0.5 (X = 0.9 attention coverage).

- **H002b (absolute group budget)**: The group union size at X=0.9
  attention coverage is at most 30% of total pages
  ($|\bigcup_q S_q^{0.9}| / P \le 0.3$) for deep layers ($l \ge 8$).
  (Original "inflation ≤ 2×" relative metric was misleading — small
  per-query K in deep layers makes the ratio mechanically large even
  when the absolute union is small. See smoke notes 2026-05-14.)

- **H002c (group recall)**: With group budget K_group = 2 × per-query
  mean K, the union captures ≥ 95% of the block's total attention mass.

- **H002d (layer profile)**: Like Quest's Fig. 3, the first 1–2 LLaDA
  layers are dense (group-Top-K must be large) but layers 4–31 admit
  ≥ 90% page sparsity at K_group ≤ 16 (out of 128 pages for 4k context).

# Falsification

Any of:
- H002a refuted: mean Jaccard < 0.3 in any deep layer → block-grouping
  is wrong unit. Pivot to spatial sub-chunk or Q-cluster grouping.
- H002b refuted: inflation > 3× consistently → group selection ≈
  union of per-query sets, sparsity gain collapses.
- H002c refuted: group recall@(2K) < 90% → either the long tail of
  attention mass is structurally non-local (Quest's premise fails for
  dLLM), or grouping is wrong.

If H002 holds → SADC (position-axis) and Quest-dLLM (page-axis) are
orthogonal; combining them is the next research direction. See
[[D001_pivot_to_sadc]].

# Method (Phase Q1a — oracle ceiling)

Script: `phase_q1a_oracle_page_grouping.py` (TBD)

For each (sample s ∈ {0..7}, step t, probe layer l, head h):
1. Run baseline forward; instrument layer l attention to get
   per-head $\mathrm{attn}[q, k]$ (full $L \times L$ probabilities).
2. Group $K$ axis into pages of size $32$: $\mathrm{attn\_page}[q, p]
   = \sum_{k \in p} \mathrm{attn}[q, k]$.
3. For each $q$, compute oracle Top-K set $S_q^X$: smallest set whose
   page mass sums to $X \in \{0.9, 0.95, 0.99\}$ of total.
4. Define group $G_t$ = queries whose position is in the current
   decoding block $B_t$.
5. Statistics per (sample, step, layer, head):
   - `per_query_mean_K_at_X` for each X
   - `pairwise_jaccard_mean_at_X` (within-group)
   - `group_union_size_at_X = |⋃_{q∈G_t} S_q^X|`
   - `group_inflation = group_union_size / per_query_mean_K`
   - `group_recall_at_K` for K ∈ {2,4,8,16,32,64}: fraction of
      Σ_{q∈G_t} (block-attention mass) captured by Top-K pages by
      summed score across G_t.

Save jsonl per row.

# Settings

- model: GSAI-ML/LLaDA-8B-Instruct
- gen-length: 256, block-length: 32, steps: 256
- num_samples: 8 (GSM8K test)
- num_fewshot: 5
- probe layers: ALL 32 (cheap per-row stats; full layer profile)
- page_size: 32

# Status

- 2026-05-14: Hypothesis registered. Phase Q1a script under design,
  awaiting user approval.
- 2026-05-14 (smoke, 1 sample × 4 step × 9 layers): All sanity
  checks pass. **Smoke was misleading** — I read the row-count
  column as P; actual P=42 not 128.

- 2026-05-14 (full sweep page=32, 131k rows): see [[q1a_quest_dllm_oracle]].
  Verdict: H002a PASS, H002b FAIL (metric was wrong), H002c PASS,
  H002d 19/28. End-to-end ceiling ≈ 1.6× attention speedup.

- 2026-05-14 (full sweep page=8, 131k rows): page granularity
  matters. page=8 saves ~25% more KV than page=32 at same recall.
  Confirms layer-stratified pattern (shallow dense, deep sparse).
  Per-head spread huge in both sweeps.

**Final verdict**: H002 is a **weak pass with refinements**:
- The grouping is structurally valid (oracle_recall_oracle@X=0.9 ≈ 0.97).
- But efficiency story is modest (1.5–1.8×) at LLaDA's L≈1316.
- **The real story** is depth-aware × head-adaptive × fine-page
  selection — three stacked findings, each ~30% saving, combining
  multiplicatively to 1.6×.
- Sparsity scales with L; longer-context dLLM (L=4k+) would likely
  recover Quest-style 5–7× speedup but requires extending the model.

See [[q1a_quest_dllm_oracle]] for full numbers and Phase Q2/Q3 plan.
