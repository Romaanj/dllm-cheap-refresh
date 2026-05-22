# Session state 2026-05-22 — window-only ablation + per-layer schedule + mechanism

Continuation of 2026-05-21 experiments. Goal: characterize the role of each component in `cheap_e3` (lag-1 top-K + U_t + window) and understand why our v3 method (LLaDA-8B-Instruct) wins on accuracy but is slower than Fast-dLLM.

## TL;DR

1. **`lag-1 top-K` is task-dependent value, not always positive**. On GSM8K, removing it (window-only) IMPROVES strict-match by +7.4pp (0.4943 → 0.5686). On HE, removing it loses −12.2pp (0.4756 → 0.3537).
2. **Per-layer attention-locality probes do NOT translate to good refresh schedules**. Data-driven Schedule A is *worse* than uniform w=4. Schedule B (wider) matches uniform w=8. Attention top-1 offset is the wrong signal for refresh allocation.
3. **Window-only mode saturates around 0.378 on HE** regardless of schedule. The 12pp gap to v3 K=384 can only be closed by global lag-1 top-K refresh — windows can't substitute.
4. **Our method's true differentiation vs Fast-dLLM**: per-step selective intra-block prefix refresh via lag-1 top-K, vs Fast-dLLM's block-boundary full forward refresh. We're slower (~5× more positions/step) but more accurate (catches intra-block prefix drift Fast-dLLM misses).
5. **Confidence freezing artifact**: positions outside W_idx in cheap_e3 are *bit-frozen* — same input → same hidden state → same logit. They commit only when their warmup-time confidence ≥0.9 or when W_idx later covers them. This implicit caching effect helps GSM8K (clean #### N format preserved) and hurts HE (no intra-block context accumulation).

## v3 baseline configuration (Ours)

`cheap_e3=True, cheap_K=384, cheap_k_schedule=stepped, cheap_warmup=2, cheap_window=4, cheap_include_u_t=True, threshold=0.9, gen=512, steps=512, block_length=32`

Stepped K schedule: `[K/2]·8 + [K]·16 + [3K/4]·8` per layer.

## Experiment 1 — K-sweep on HE (GPU 0, chained)

cheap_e3 full with window=4 + lag-1 top-K, varying K.

| K | cleaned pass@1 | wall (s) | speedup | NFE/sample |
|---|---|---|---|---|
| 384 (baseline) | 0.4756 | 2361 | 2.95× | — |
| 256 | 0.4268 | 2010 | 3.47× | 236.5 |
| 192 | 0.4207 | 1941 | 3.59× | 236.8 |
| 128 | 0.4329 | 1818 | 3.83× | 240.2 |
| 64 | 0.3598 | 1719 | 4.05× | 241.3 |

**Findings**:
- K=384 is clearly best. K reduction monotonically hurts accuracy.
- Wall savings are modest (1.17-1.41×): bottleneck is NFE (steps), not per-step attention compute.
- NFE is constant ~240 across all K → θ=0.9 commit aggression doesn't change with K.

## Experiment 2 — v4 window-only (GPU 0, then GPU 1)

`cheap_window_only=True`: build_E3_A_per_layer returns only the window set, no lag-1 top-K fill. K parameter is ignored.

**Code change**: added `window_only: bool = False` parameter to `phase14a_cheap_e2e.build_E3_A_per_layer`, `cheap_e3_inference_llada.generate_cheap_e3`, and `eval_llada.cheap_window_only`. ~10 lines.

| Task | Config | pass@1 / metric | wall | vs v3 K=384 |
|---|---|---|---|---|
| HE | v4 window=4 | 0.3537 | 1669s | −12.2pp, +1.41× speed |
| GSM8K | v4 window=4 | flex 0.7149, **strict 0.5686** | 8742.8s | flex −4.6pp, **strict +7.4pp**, +1.26× speed |

**Surprising finding**: GSM8K strict-match improvement is statistically significant (3.8σ over v3, +16pp over vanilla). Window-only outperforms v3 AND vanilla on strict — first sign that "more refresh" is not monotonically better.

## Experiment 3 — |U_t| / |W_idx| measurement (GPU 1, instrumented probe)

Monkey-patched `phase14a_cheap_e2e.window_set_indices` to record per-step `|U_t|` and `|W_idx|`. Ran 5 HE samples with v4 window-only.

| stat | \|U_t\| (commits/step) | \|W_idx\| (refresh/layer/step) |
|---|---|---|
| mean | 1.97 | 10.16 |
| median | 1 | 9 |
| p90 | 4 | 12 |
| max | 11 | 24 |

- Dedup factor: 6.94 / 9 = 0.77 (commits cluster spatially)
- v4 refreshes **~3.3% of v3's budget per layer per step** (10/312)
- v4 refreshes **~1.5% of full attention coverage** (10/650)

## Experiment 4 — Per-layer window schedule (GPU 1, chained)

**Motivation**: probe data in `results_fastgen_head_probe/mask_binder_self_vs_neighbor/agg.npz` measures per-layer top-1 attention offset distribution. Hypothesis: layers with more concentrated local attention need smaller w, layers with distributed attention need larger w.

**Per-layer locality** (`|off|≤4` capture fraction):
- Early (L0-L9): 30-50% (distributed)
- Middle (L10-L19): 24-51% (mixed)
- Late (L20-L31): 47-69% (highly local)

Counter to existing stepped K schedule (mid-heavy), data suggests early-heavy allocation.

**Code change**: extended `cheap_window` parsing to accept int OR `;`-separated list (lm-eval splits model_args on `,` so list separator must be `;`). Propagated `window_per_layer` through `generate_cheap_e3` to `build_E3_A_per_layer`. ~10 lines.

**Schedules tested** (all with `cheap_window_only=True`, K ignored):

| Schedule | desc | avg w | HE pass@1 | wall | vs v4 uniform w=4 |
|---|---|---|---|---|---|
| A (target 60% capture) | early ↑, late ↓ (w=3 at L28/30/31) | 8.25 | 0.3232 | 1845s | **−3.0pp** |
| B (target 70% capture) | early ↑↑, late ≥5 | 11.75 | 0.3780 | 1573s | +2.4pp |
| uniform w=8 | flat | 8.0 | **0.3780** | **1374s** | +2.4pp |

**Findings**:
- Schedule A (data-driven, same avg w as uniform w=8) is *worse* than uniform — late-layer w=3 was catastrophic.
- Schedule B (more compute) ties with uniform w=8 (less compute) — per-layer info adds nothing.
- **Negative result for paper**: attention top-1 offset is NOT the right signal for refresh allocation.
- Window-only family saturates at ~0.378 on HE regardless of schedule.

## Mechanism understanding (after reading sparse_block.py)

### What "refresh" actually means in cheap_e3

For positions in `W_idx` (= A_l):
- **K/V cache** updated at `A_input_changed` (= U_t for layer 0, then prev layer's A_l for layer ≥ 1 — cascade grows)
- **Q computed** at A_l only (sparse attention queries)
- **FFN + residual** at A_l only
- **Hidden state output** written back at A_l only; non-A_l positions inherit `prev_block_out` (bit-identical)

For positions NOT in any A_l across the whole step:
- Hidden state bit-frozen at warmup-time (or whenever last refreshed)
- Therefore logit bit-frozen, confidence bit-frozen
- **Commit only if warmup confidence ≥ θ, or when later W_idx covers them**

### vs Fast-dLLM dual-cache (NVlabs/Fast-dLLM v1/llada/generate.py:211)

Fast-dLLM `generate_with_dual_cache`:
```python
for nb in range(num_blocks):
    out_full = model(x, use_cache=True)         # FULL FORWARD (T positions) at block start
    past_key_values = out_full.past_key_values
    nfe += 1
    # initial transfer using full logits
    for i in range(1, steps_per_block):
        logits_blk = model(x[:, s:e], past_key_values=past_key_values, replace_position=...)
        # block-only forward with cached prefix/suffix K/V
        nfe += 1
```

**Architecture**:
- Cache regions: prefix (positions < s, frozen during block), suffix (positions > e, frozen during block), block (s..e-1, replaced via `replace_position`)
- Block boundary: 1 full forward refreshes both caches with current x state
- Intra-block: 32 Q × full T K/V (cached), only block K/V replaced

**vs Ours v3**:
| | Fast-dLLM | Ours v3 |
|---|---|---|
| Block boundary | full forward (T positions × n_layers) | 0 (no separate handling) |
| Per-step within block | 32 Q + 32 K/V refresh | ~312 Q + cascade K/V at A_per_layer |
| Total per block | T + ~31×32 ≈ T + 1000 | 16 × 312 ≈ 5000 |
| Prefix intra-block freshness | step 0 fresh, then stale | per-step lag-1 selective refresh |

Fast-dLLM HE: 0.4329, 1278s, 5.45×
Ours v3 HE: 0.4756, 2361s, 2.95×

→ Ours pays 5× more per-step compute for +4.3pp accuracy. The advantage is *intra-block prefix refresh* (catching drift Fast-dLLM ignores between block boundaries).

→ **v4 is strictly dominated by Fast-dLLM on HE** (both slower AND less accurate: 4.18× vs 5.45×, 0.3537 vs 0.4329) — confirms lag-1 top-K is what gives v3 its Pareto advantage.

## Identified structural issue in window-only mode (discussion only, not tested)

**Issue raised in conversation**: in current block, positions outside `U_t ± w` but still within block are:
- Not in W_idx → hidden state frozen → logit frozen
- Their commit decision uses warmup-time logits, never updated

For HE (code with global dependencies): these positions need fresh logits as block context evolves → catastrophic 12pp loss.
For GSM8K (local math reasoning): warmup logits are already good for these positions → window-only actually preserves cleaner output format.

**Discussed but not implemented**:
- *Option A* — force W_idx ⊇ all current-block masks (Fast-dLLM-like full block refresh): wastes prompt-side window, makes us dominated by Fast-dLLM
- *Option B* — "shadow query" only at last layer for block masks outside W_idx: compute fresh logit without updating residual stream (preserves "old info" while getting fresh confidence)

## Decision log

- **Not implementing block-full-refresh hybrid (Option A)**: would lose our differentiation; we'd become Fast-dLLM plus wasted prompt-side compute. Per user: "그냥 fast-dllm이랑 뭐가 다른거지? 오히려 더 많은 계산량만 쓰는거 아니야?"
- **Shadow-query (Option B) parked**: more complex (~50 lines, needs Q-only sparse variant); evaluate after MBPP results land.
- **Per-layer window scheduling: parking direction**. Negative result clear: attention top-1 offset doesn't help. If revisited, would need a different signal (drift sensitivity, occlusion experiments).

## Pending experiment (running at session end)

**MBPP v4 window-only** on GPU 0, started 2026-05-22T08:49Z. At session end (~09:35Z): 407/500 samples done (~81%, ETA ~18 min).
- Script: `/tmp/v4_window_only_mbpp_gpu0.sh`
- Chain log: `/tmp/v4_window_only_mbpp_chain.log`
- Output: `results_lmeval_dream_cheap/llada_ours_mbpp_3shot_g512_chaton_v4_window_only/`
- Postprocess: `python postprocess_mbpp.py <samples.jsonl>`

**Hypothesis test**:
- v3 K=384 MBPP: cleaned pass@1 = 0.368 (from `/tmp/mbpp_ours_elastic_chain.log` 2026-05-21)
- Fast-dLLM MBPP: cleaned pass@1 = 0.382 (from same chain)
- If v4 catastrophic loss (like HE): "code task = global dep, dual-mechanism needed" confirmed
- If v4 OK or improves (like GSM8K strict): MBPP is more "local-structured" than expected

## Key code paths

- `/workspace/inverse_cdf/Fast-dLLM/llada/phase14a_cheap_e2e.py:173` — `build_E3_A_per_layer` (with `window_only` flag)
- `/workspace/inverse_cdf/Fast-dLLM/llada/cheap_e3_inference_llada.py:32` — `generate_cheap_e3` (window can be int or list)
- `/workspace/inverse_cdf/Fast-dLLM/llada/eval_llada.py:90-95,245-258` — model_args parsing including `cheap_window_only`, `cheap_window` list (`;` separator due to lm-eval `,` collision)
- `/workspace/inverse_cdf/Fast-dLLM/llada/sparse_block.py:402` — `sparse_hybrid_cascade_forward_with_kv` (K/V cache cascade, A_input_changed = prev layer's A_l)
- `/workspace/inverse_cdf/Fast-dLLM/llada/baseline/Fast-dLLM/v1/llada/generate.py:211` — Fast-dLLM `generate_with_dual_cache` reference

## Result file locations

- HE K-sweep K=64..256: `results_lmeval_dream_cheap/llada_ours_humaneval_g512_chaton_K{64,128,192,256}/`
- HE v4: `results_lmeval_dream_cheap/llada_ours_humaneval_g512_chaton_v4_window_only/`
- HE v5 schedules: `results_lmeval_dream_cheap/llada_ours_humaneval_g512_chaton_v5_{schedA_target60,schedB_target70,uniform_w8}/`
- GSM8K ablations (v1/v2/v3): `results_lmeval_dream_cheap/llada_ours_ablation_v{1_topk_only,2_topk_ut,3_full}/`
- GSM8K v4 window-only: `results_lmeval_dream_cheap/llada_ours_ablation_v4_window_only/`
- |W_idx| probe results: `/tmp/w_idx_probe_results.json`

## Related

- [[universal_sink_window_abstraction]] — head-level sink+window taxonomy used to derive Schedule A/B
- [[phase13b_cheap_estimator]] — original cheap_e3 design notes
- [[phase14a_e2e_results]] — initial v3 benchmark results
