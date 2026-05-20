# Research index

_Auto-maintained by /lit-search and /synthesize._

## Literature

Searched for prior work on mask-token attention in dLLMs (re: `mask_binder` finding). Verdict: **partial novelty**; MaskKV is closest prior art, but no published per-head clustering with cross-task IoU on LLaDA.

- [MaskKV — Mask Tokens as Prophet (2510.09309)](lit/maskkv_2510_09309.md) — **closest prior art**. Distinguishes "Information" vs "Structure" heads via mask-vs-prompt attention ratio; uses signal for prompt-KV eviction. Does NOT report head-population fraction, layer concentration, or cross-task IoU.
- [Attention Sinks in DLMs (2510.15731)](lit/attention_sinks_dlm_2510_15731.md) — first sink characterization in DLMs. **Explicitly: LLaDA sinks land on punctuation/whitespace, NOT mask** — different head population than our `mask_binder`. Dream-7B sinks shift on masked positions (closest phenomenology).
- [Elastic-Cache (2510.14973)](lit/elastic_cache_2510_14973.md) — claims "distant MASK = length-bias" (aggregate); contradicted by our 20% subpopulation. Good positioning foil.
- [DPad — Suffix Dropout (2508.14148)](lit/dpad_2508_14148.md) — aggregate attention to distant suffix is small; no per-head analysis.
- [Fast-dLLM (2505.22618)](lit/fast_dllm_2505_22618.md) — cache infrastructure; no head-level analysis.
- [DLM-Scope (2602.05859)](lit/dlm_scope_2602_05859.md) — SAE features in LLaDA/Dream; no attention-head analysis.
- [Sink Token (2601.19657)](lit/sink_token_2601_19657.md) — training-side sink stabilization; orthogonal.
- [SparseD (2509.24014)](lit/sparsed_2509_24014.md) — closest method-side prior. Per-head sparse attention for dLLM (LLaDA + Dream). Observes head heterogeneity + temporal stability but **no named taxonomy, no fractions, no cross-model alignment**.
- [d2Cache (2509.23094, ICLR'26)](lit/d2cache_2509_23094.md) — adaptive per-**token** KV cache for dLLM; isotropic Gaussian window in sequence position (σ=10), uniform across heads. KV-level reuse, not attention-output reuse.
- [EntropyCache (2603.18489)](lit/entropycache_2603_18489.md) — recompute KV for `currently masked ∪ k=64 most recently unmasked`; recency proxy, not per-head scope. Closest to "event-driven on newly unmasked" but globally and at KV granularity.
- [RetroAttention (2508.09001)](lit/retroattention_2508_09001.md) — **AR-only** attention-OUTPUT cache; retrospectively updates past Queries when new KV pages arrive. Per-head "last-loaded step" mask. Same skeleton as our plan but AR + recency-scope.
- [ELF — Embedded Language Flows (2605.10938)](lit/elf_2605_10938.md) — **different family**: continuous Flow Matching DLM (MIT/Kaiming He). Trajectory entirely in T5-embed space; discretization only at t=1 via shared-weight "decode mode". No remask schedule → inverse-CDF/cache work has no surface. Concurrent: LangFlow, FLM, CoDAR.
- [SPA-Cache (2602.02544)](lit/spa_cache_2602_02544.md) — **closest prior art** (concurrent, Jan 2026). Per-(layer × position) refresh-only caching using SVD-truncated Value proxy + cosine similarity vs cached proxy. Same paradigm as our Phase 12-14. Up to 8× throughput.
- [Window-Diffusion (2601.20332)](lit/window_diffusion_2601_20332.md) — 99× speedup, but mostly from suffix pruning + EOS-early-stop; geometric window uniform across layers; phase-based (every 32 steps) refresh.
- [dInfer (2510.08666)](lit/dinfer_2510_08666.md) — system inference framework; vicinity geometric refresh; 1100 TPS HumanEval; 10× over Fast-dLLM.
- [dKV-Cache (2505.15781)](lit/dkv_cache_2505_15781.md) — delayed/conditioned KV caching (cache KV one step AFTER decode); reuse-based; 2-10× speedup.

## Experiments

- phase 11 oracle active-set ceiling — `results_phase11_oracle/oracle_active_set.jsonl` (running, job ID `b8nz3a5uf`)
- phase 12 cascade oracle (per-layer × position refresh) — `results_phase12_cascade_oracle/gpu{0,1}/cascade_oracle.jsonl` (done, 1200 rows, see [[phase12_cascade_oracle]])
- phase 13a oracle position geometry — `results_phase13a_geometry/gpu{0,1}/oracle_positions.jsonl` (done, 2040 rows, see [[phase13a_oracle_geometry]])
- phase 13b cheap estimator — `results_phase13b_cheap/gpu{0,1}/cheap_estimator.jsonl` (done, 3000 rows, see [[phase13b_cheap_estimator]])
- phase 14a GSM8K end-to-end — `results_phase14a_e2e/{baseline,cheap_e3}_n100/` (done, 100 each, see [[phase14a_e2e_results]])
- phase 14a HumanEval end-to-end — `results_phase14a_he/{baseline,cheap_e3}/` (done, 164 each)
- FastGen-style per-head attention pattern probe on LLaDA (3 GSM8K × gen=128) — `results_fastgen_head_probe/medium/`
- Q1a oracle page-grouping (page=32) — `results_q1a_oracle/page_grouping.jsonl` (done, 131k rows)
- Q1a oracle page-grouping (page=8)  — `results_q1a_oracle_p8/page_grouping.jsonl` (done, 131k rows)

## Notes

- [FastGen head-pattern probe on LLaDA](notes/fastgen_head_pattern_probe.md) — 5 FastGen patterns collapse to {SMP, SMPF, FULL}; drift is block-quantized → per-block profiling is sufficient (within-block mode-fraction 0.93 vs global 0.86)
- [Q1a — Quest-dLLM oracle ceiling (page=32 + page=8)](notes/q1a_quest_dllm_oracle.md) — depth-stratified, per-head heterogeneous sparsity. Oracle ceiling ≈ 1.6× attention speedup at L=1316. Layer-stratified pattern matches Quest's AR-LLM finding. **Relates to [[fastgen_head_pattern_probe]]** — both surface per-head heterogeneity from independent angles.
- [Universal sink+window abstraction for MDM heads](notes/universal_sink_window_abstraction.md) — proposed cross-LLaDA-Dream abstraction: every head = sink_set ∪ local_window. Sink identity & window size are model-specific; the dichotomy is universal. Basis for sparse attention method.
- [R26 SCOPE-DRIVEN INCREMENTAL: negative result + diagnosis](notes/R26_scope_incremental_negative_result.md) — cache-reuse methods (static scope V1/V2 + dynamic drift V0) fundamentally fail for dLLM. Mechanisms: bidir attention cascading (r=0.86 cross-layer), peak shift mobility (57%), diffuse attention. R_curr's 0.755 remains best accessible. Includes future path β suggestions.
- [Phase 12 — Cascade-respecting per-(layer × position) refresh oracle](notes/phase12_cascade_oracle.md) — **positive complement to R26**. Refresh-only paradigm (compute A_l fresh, reuse stale only outside A_l) gives cos 0.998 at K=128 (10% L), argmax 0.97 at K=384 (28% L). Smooth concave curve, no cliff. H004 strongly supported. Deep layers have MORE active positions than shallow (cascade accumulation), not fewer.
- [Phase 13a — Oracle active-set geometry](notes/phase13a_oracle_geometry.md) — A_oracle is NOT a simple geometric function of U_t. Median distance 150-450; 50-77% of A_oracle lives in PROMPT (bidir attention cascade). Window-only estimator caps at IoU 0.41.
- [Phase 13b — Cheap (lag-1) estimator](notes/phase13b_cheap_estimator.md) — **method viable**. Lag-1 history captures 80%+ of oracle gain at K≥256. E3 (U+window4+lag-1) hits argmax 0.952 at K=384 (28% L). Cheap-vs-oracle penalty ≈ 1.5× budget. Memory cost: one extra snapshot.
- [Phase 14a — End-to-end accuracy results](notes/phase14a_e2e_results.md) — **baseline parity confirmed**. GSM8K n=100: 0.810 (base) vs 0.820 (cheap K=384). HumanEval n=164: 0.372 vs 0.384 (K=128). Both gaps within ±SE. 254 consecutive cheap forwards don't degrade quality.
- [Phase 12 (Dream-7B) — Cascade oracle port](notes/phase12_cascade_oracle_dream.md) — **Dream's cascade containment ≥ LLaDA's** (+1–5pp argmax at every K). K=128 arg 0.963, K=384 arg 0.982. Dream cheap mode -10pp drop is NOT explained by oracle ceiling — bottleneck is lag-1 estimator or KV-cache mechanism. Deep layer concentration is HIGHER than LLaDA's → uniform K may suit Dream better than stepped.
- [Novelty positioning 2026-05-18](notes/novelty_positioning_2026_05_18.md) — comparative analysis after lit-search. SPA-Cache (Jan'26) closest prior; refined claim: lag-1 history signal + cascade containment mechanism + planned masked kernel as distinguishing contribution.
- [**Session state — 2026-05-20** (handover)](notes/session_state_2026_05_20.md) — full state of paper-comparison work (Fast-dLLM, Dynamic-DLLM, Elastic-Cache vs Ours) on LLaDA+Dream; current numbers, in-flight jobs, code diagnoses (APD threshold drift = main Dynamic-DLLM degradation cause), open decisions (D1: new GitHub repo, D2: Pareto framing, D3: add MBPP/MATH-500). **Read this first** when resuming.

## Hypotheses

- [H001 — Active-set compressibility of deep ΔH](hypotheses/H001_active_set_compressibility.md) — open
- [H002 — Query-group compressibility of KV-page attention](hypotheses/H002_query_group_page_compressibility.md) — weak-pass with refinements (2026-05-14)
- [H003 — What is mask_binder computing?](hypotheses/H003_mask_binder_computation.md) — open exploration (2026-05-17)
- [H004 — Cascade-respecting per-(layer × position) active set](hypotheses/H004_cascade_active_set.md) — open (2026-05-18); user's core question reframed

## Decisions

- [D001 — Pivot from Path A to SADC](decisions/D001_pivot_to_sadc.md) — chosen 2026-05-13

- [D002 — HumanEval protocol: chat ON + gen=512 unification](decisions/D002_chat_template_and_gen_length_unification.md) — 2026-05-20
