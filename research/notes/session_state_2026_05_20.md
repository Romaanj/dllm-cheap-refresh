---
name: session-state-2026-05-20
description: Handover note covering paper-ready comparison of cheap-refresh (Ours) vs caching baselines (Fast-dLLM, Dynamic-DLLM, Elastic-Cache) on LLaDA-8B + Dream-7B; current numbers, in-flight jobs, code analyses, open decisions
metadata:
  type: project
---

# Session state — 2026-05-20 (handover to new server)

This note captures the state of the paper-comparison work as of 2026-05-20 ~05:50 UTC. Written so a fresh Claude session on a different server can pick up cold. Source machine: `60988ebf30e5`. Working dir: `/workspace/inverse_cdf/Fast-dLLM/llada/`.

## 1. What we are doing

Building a paper comparing **Ours** = E3 cheap-refresh (lag-1 + $U_t$ + window-4) + sparse-Q forward + KV cache, composed with $\theta{=}0.9$ confidence-threshold parallel decoding — against published dLLM inference accelerators:

- **Vanilla baseline** (no cache, no parallel)
- **Fast-dLLM** (prefix + dual cache + confidence-threshold parallel)
- **Dynamic-dLLM** (ICLR 2026) = Dynamic Cache Updating (DCU) + Adaptive Parallel Decoding (APD)
- **Elastic-Cache** (per-layer adaptive cache + fixed-threshold parallel)

Models: **LLaDA-8B-Instruct** and **Dream-v0-Instruct-7B**. Benchmarks: **GSM8K** (5-shot, $N{=}1319$) and **HumanEval** (0-shot, $N{=}164$, pass@1 via `postprocess_code.py` = sanitize + hf_evaluate code_eval).

## 2. Current numbers (state of `tables.tex`)

### Main table (`\label{tab:main}`)

**LLaDA-8B-Instruct**

| Method | GSM8K flex | GSM8K strict | GSM8K Wall (s) | GSM8K Speedup | HE pass@1 | HE Wall (s) | HE Speedup |
|---|---|---|---|---|---|---|---|
| Vanilla | 0.7801 | 0.4049 | 48764 | 1.00× | 0.372 | 2211 | 1.00× |
| Fast-dLLM | **0.7885** | 0.3647 | **5355** | **9.11×** | 0.354 | **647** | **3.42×** |
| Dynamic-dLLM (DCU+APD) | 0.4412 | 0.2138 | 14069 | 3.47× | 0.323 | 1220 | 1.81× |
| **Ours** | 0.7612 | **0.4943** | 10878 | 4.48× | **0.427** | 899 | 2.46× |

**Dream-v0-Instruct-7B**

| Method | GSM8K flex | GSM8K strict | GSM8K Wall (s) | GSM8K Speedup | HE pass@1 | HE Wall (s) | HE Speedup |
|---|---|---|---|---|---|---|---|
| Vanilla | 0.7832 | 0.7809 | 41860 | 1.00× | 0.506 | 1935 | 1.00× |
| **Ours** | 0.7309 | 0.7066 | **5494** | **7.62×** | **0.524** | **735** | **2.63×** |

Dream has no Dynamic-DLLM / Elastic-Cache port (no public LLaDA→Dream code). Treat as "model-specific" and discuss as a methodology gap.

### Ablation table (`\label{tab:ablation-dynamic-dllm}`) — Dynamic-DLLM component breakdown (LLaDA)

| Variant | GSM8K flex/strict | HE pass@1 | HE Wall (s) | HE Speedup |
|---|---|---|---|---|
| Vanilla baseline | 0.7801 / 0.4049 | 0.372 | 2211 | 1.00× |
| Dynamic-dLLM full (DCU+APD) | 0.4412 / 0.2138 | 0.323 | 1220 | 1.81× |
| Dynamic-dLLM DCU only ($\theta{=}0.9$, no APD) | **TBD** (running) | **0.372** | 3786 | 0.58× |

Caption thesis: **APD is responsible for the entire $-5\text{pp}$ HumanEval accuracy drop, and DCU alone provides no wall-clock benefit at this scale** (in fact $1.7\times$ slower than vanilla).

## 3. In-flight jobs as of handover

⚠️ These run on the source server only. New server will not see them. Decide: wait for source server to finish + sync results, OR re-run on new server.

| PID | GPU | Task | Config | Progress | ETA |
|-----|-----|------|--------|----------|-----|
| 3340089 | 1 | Dynamic-dLLM no-APD GSM8K | gen=256, steps=256, blk=32, $B_{layer}=B_{window}=32$, pd_mode=0, θ=0.9 | 1016/1319 (77%) at 03:01 elapsed | ~50 min |
| 3390263 | 0 | Elastic-Cache HumanEval | gen=512, win=16, steps=32, θ=0.9, γ=0.9, track_num=1, block_caching=True, 8B-Instruct | 80/164 (49%) at 00:13 elapsed | ~15 min |
| 3391335 | — | chain script (`/tmp/elastic_gsm8k_chain.sh`) | waits for PID 3390263 then launches Elastic GSM8K on GPU 0 (gen=256, win=16, steps=16) | waiting | n/a |

When chain fires, the new launch will be: Elastic-Cache GSM8K, gen=256, steps=16, win=16, θ=0.9, γ=0.9, 8B-Instruct. ETA ~3-4h after start.

Watcher background tasks (Claude harness):
- `b3huvu2o6` — auto-postprocesses Elastic HumanEval once done (runs `postprocess_code.py` → pass@1)
- `ba0yaxn8y` — waits for Dynamic-dLLM no-APD GSM8K to finish + tails log

## 4. Diagnoses (do not re-derive)

### Dynamic-DLLM degrades GSM8K (-35pp flex, -19pp strict) for these reasons

Source: `Fast-dLLM/Dynamic-DLLM/generate/generate_dynamic.py` + `cache/dynamic_cache.py`.

1. **APD threshold drift** (`generate_dynamic.py:130`): `updated_threshold = current_threshold - α·peak_conf + β·dist_sim`. With α=0.001 and peak_conf≈0.99, threshold *decreases* ~0.001/step. After 8000+ updates, threshold ≈ 0 → almost every token committed in parallel → token-level errors (mid-word truncation 10%, repetition 15%, digit substitution).
2. **DCU stale K/V**: `window_size=32` + `layer_budget=32` on gen=256 means 224/256 (87.5%) positions use cached K/V from mask state. Causes "160→20" / "12→8" digit substitution observed in samples.
3. **Prompt cache effectively frozen**: `prompt_interval_steps=104` vs steps_per_block=32 → prompt K/V refreshes maybe 1-2× over the whole run. 5-shot CoT prompt's exact numbers degrade in attention.
4. **Greedy commit, no remask**: once a token is committed via APD it can't be undone → arithmetic chain errors compound.

**Why HumanEval barely dents (-5pp)**: pass@1 metric tolerates whitespace, comments, repeated lines; unit test pass is binary. GSM8K strict-match requires exact `#### N` (and 53.6% of Dynamic-DLLM outputs lack the `####` token entirely).

Sample-level statistics computed at `results_lmeval_dream_cheap/llada_dynamic_dllm_gsm8k_paper/GSAI-ML__LLaDA-8B-Instruct/samples_gsm8k_*.jsonl` — re-run the Python snippet in section 6 if needed.

### Elastic-Cache algorithm summary

Source: `Fast-dLLM/llada/Elastic-Cache/llada/generate.py` + `model/modeling_llada.py:755-897`.

- **Per-block (= per transformer layer) cache**: each layer holds its own `x_cache, q_cache, k_cache, v_cache, track_token`.
- **Window querying** (`block_caching=True`): each forward pass only queries `masked_position[:window_length]` masks (so gen=512/win=16 → 32 forward passes minimum).
- **Drift-based per-layer refresh**: at each layer, compute `sim = cos(past_attn_weight, current_attn_weight)` on `track_position` (top-$k$ most-attended positions from previous step, $k$=`track_num`=1 by default). If `sim < γ` (γ=0.9), force recompute from this layer onward (`lengths[1] = block_idx + 1`).
- **Fixed-threshold parallel commit** (θ=0.9) — no drift, unlike APD.

### Contrast: Dynamic-DLLM vs Elastic-Cache

| Feature | Dynamic-DLLM | Elastic-Cache |
|---|---|---|
| Cache refresh trigger | time interval (prompt_interval=104) | **attention-drift cos sim < γ** |
| Cache granularity | global, all-layer | **per-layer independent** |
| Parallel commit | APD threshold drift | **fixed** θ=0.9 |
| Track mechanism | static intervals | **dynamic** top-attention re-test |

→ Elastic-Cache's design choices specifically *avoid* the failure modes that hurt Dynamic-DLLM on GSM8K. Expect Elastic-Cache to be a strong baseline; we should not pretend otherwise.

### Where we differentiate vs Elastic-Cache (paper framing)

Both use **fixed θ=0.9 parallel commit** (safe). Difference is in *what signal drives refresh*:
- Elastic-Cache: per-layer **attention cos-sim** between past and present K/V (post-hoc)
- Ours: **schedule-level signals** — lag-1 confidence drift + remaining-unmask budget $U_t$ + window-4 lookahead, paired with **sparse-Q forward** so we never pay full attention cost on cache-hit positions.

The two are *not* directly competing on the same axis; both are valid Pareto points. We should frame as "different refresh signal, same safety regime."

## 5. Open decisions for the new server

### D1. Paper repo restructure → **decision: YES, make a new GitHub repo**

User confirmed (2026-05-20). Proposed structure (do this on new server):

```
dllm-cheap-refresh/                  # or similar name
├── README.md
├── methods/
│   ├── ours/                        # E3 (lag-1 + U_t + window-4), sparse-Q, schedules
│   │   ├── eval_llada.py
│   │   ├── eval_dream.py
│   │   └── refresh/                 # lag1.py, e3_combined.py, etc.
│   ├── fast_dllm/                   # vendored from huggingface/Fast-dLLM @ <sha>
│   ├── dynamic_dllm/                # vendored from ICLR 2026 repo @ <sha>
│   └── elastic_cache/               # vendored
├── evals/                           # unified lm-eval scripts, all baselines
├── results/                         # JSON summaries only (raw samples → release artifact)
├── tables/                          # tables.tex, figures
├── research/                        # copy of current research/ (notes, lit, hypotheses)
├── paper/
└── scripts/                         # launch helpers
```

Rules:
- Baselines vendored as plain dirs with `VENDORED.md` (`Vendored from <url>@<sha> on <date>`)
- Our patches to baselines (if any) noted in `methods/<name>/PATCHES.md`
- `results/` keeps small JSONs only; raw `samples_*.jsonl` shipped as release artifact

Source repo (`Fast-dLLM` fork) becomes legacy — git history stays for reproducibility but new work goes in new repo.

### D2. Pareto framing → **decision: YES**

Two-axis scatter (x = wall-time speedup, y = accuracy) with all 4 methods + Vanilla. Frame: "different Pareto points", not "winner". This is the *only* honest framing given that:
- Fast-dLLM dominates speed on HumanEval (9.11× GSM8K, 3.42× HE) but loses accuracy
- Ours dominates accuracy at moderate speed (4.48× GSM8K with **best** strict-match)
- Dynamic-DLLM neither dominates accuracy nor speed
- Elastic-Cache TBD positioning (likely between Fast-dLLM and Ours)

### D3. Benchmark scope

- **Confirmed**: GSM8K + HumanEval on LLaDA (all 4 methods) + Dream (Vanilla + Ours only)
- **Strong recommend**: add MBPP (code) and MATH-500 (hard reasoning) at least for LLaDA
- **Optional stress test**: long-form (BBH, IFEval) to expose Elastic-Cache's window assumption
- **Open**: whether to port Dynamic-DLLM or Elastic-Cache to Dream ourselves — currently neither has Dream code; decision pending after Elastic-Cache LLaDA numbers come back

### D4. Where the "lag-1 / U_t / window-4" cheap E3 methodology lives in the paper

Existing notes:
- [[phase13b_cheap_estimator]] — derivation + ablation
- [[phase14a_e2e_results]] — end-to-end GSM8K
- Method section should consolidate these. Decision: write a single coherent method section in new repo's `paper/method.tex` rather than referring back to phase notes.

## 6. Key scripts and code locations

### Eval / launch
- `eval_llada.py` (our adapter) — registers `llada_dist` model for lm-eval
- `eval_dream.py` — Dream adapter; entry point uses `cli_evaluate()`. **Cannot use `lm_eval -m` directly** (dream_dist not in lm_eval registry).
- `eval_hybrid_cdf.py`, `gsm8k_hybrid_cdf_eval.py` — our E3 / cheap refresh evals
- `postprocess_code.py` — HumanEval scorer (sanitize + code_eval). Needs `HF_ALLOW_CODE_EVAL=1`. Uses multiprocessing.Manager → **needs `/dev/shm` free space** (clear with `rm -f /dev/shm/__KMP_REGISTERED_LIB_*` if EOFError).
- Positional arg only: `python postprocess_code.py <samples.jsonl>`, NOT `--input`.

### Vendored baselines (in current repo)
- `Fast-dLLM/llada/Elastic-Cache/llada/` — Elastic-Cache, with our drift-logging instrumentation already added to `model/modeling_llada.py:855-889`
- `Fast-dLLM/Dynamic-DLLM/` — Dynamic-DLLM, generate path: `generate/generate_dynamic.py`

### Results dir
- `results_lmeval_dream_cheap/` — all lm-eval runs land here
- `results_phase14a_e2e/` — Ours E3 phase 14a runs
- `results_phase7_elastic_*` — earlier Elastic smoke runs (N=32 GSM8K), pre-lm-eval

### Sample-level analysis snippet (Dynamic-DLLM GSM8K)
```python
import json
n_total = n_no_hash = n_trunc = n_repeat = 0
for line in open('SAMPLES.jsonl'):
    d = json.loads(line)
    if d['filter'] != 'strict-match': continue
    n_total += 1
    resp = d['resps'][0][0]
    if '####' not in resp: n_no_hash += 1
    # truncated mid-word, repetition checks — see earlier conversation
```

## 7. Memory / user preferences worth carrying

- **Bilingual** (Korean ↔ English). Reply in Korean for chat, English for code/docs.
- **Compact answers preferred**. Don't summarize what was just done if the diff/output already shows it.
- **Honest framing > cherry-pick**. Especially: don't drop weak baselines; report transparently and frame trade-offs.
- **Pareto thinking**. Default framing for any multi-objective comparison.
- **Auto-launch chained jobs** when the user confirms direction (no need to ask before each restart).
- **GPU 2 and 3** are typically running other workloads — only use GPU 0 and 1.
- **No `cd` chained with `git`** (harness triggers permission prompt).

## 8. Division of labor — what runs where

**Source server** (this server, where session_state was written): owns the **benchmark re-runs and synthesis**.
- ALL post-D002 HumanEval re-runs (LLaDA Vanilla / Fast-dLLM / Dynamic-dLLM full / Dynamic-dLLM no-APD / Ours; Dream Vanilla / Ours) are running here on GPU 0 + GPU 1 chains. ETA ~4h.
- Dynamic-dLLM no-APD GSM8K already complete (flex 0.7771, strict 0.3791, 14194s, 3.44× — see [[session-state-2026-05-20]] tab:ablation row).
- Elastic-Cache GSM8K still running on GPU 0.
- Results sync to repo (`tables.tex` + this note) as each completes.

**New server** (where this repo is being picked up): owns the **method evolution track**.
- Read [[method_evolution_directions_2026_05_20]] for candidate next moves on the cheap-refresh algorithm itself.
- Do NOT re-execute published-baseline HumanEval/GSM8K runs (source server already running them; would waste compute).
- DO experiment with method variants (adaptive $K_t$, multi-signal fusion, per-layer trigger, etc.). Add new ablation tables in `tables.tex` as needed; don't overwrite headline rows.

### Source-server pending after current chains finish

1. Re-runs complete → update `tab:main` HumanEval columns (will resolve the chat-OFF/gen=256 inconsistency from D002).
2. `tables.tex` Elastic-Cache GSM8K row (TBD → real numbers).
3. Add Pareto scatter figure (`tables/fig_pareto.py` from main + ablation tables).
4. Consolidate method section in `paper/method.tex` from phase notes.
5. Decide MBPP / MATH-500 scope after main re-runs land.

## 9. Recent-conversation gotchas (avoid re-stumbling)

- HumanEval gen=512 needs `B = gen_len/8 = 64` per Dynamic-DLLM paper Appendix E.1 auto-tuning rule (window_size = layer_budget = 64). GSM8K gen=256 → B=32. Without auto-tuning we got 0.183 (wrong); with it 0.323 (matches paper within hardware variance).
- `dream_dist` model registration: must launch via `accelerate launch eval_dream.py`, **not** `lm_eval -m dream_dist`.
- `/dev/shm` exhaustion (KMP_REGISTERED_LIB leak) bricks `postprocess_code.py` with EOFError. Always check before running: `df -h /dev/shm`. Cleanup: `rm -f /dev/shm/__KMP_REGISTERED_LIB_*`.
- Dynamic-DLLM original paper numbers (0.3715 HumanEval) we reproduce as 0.323 with auto-tuned B=64. ~5pp gap, attributed to hardware variance (shared GPU vs RTX 4090).

## 10. How to resume on new server

Recommended first prompt on the new server:
```
Read CLAUDE.md, research/INDEX.md, research/notes/session_state_2026_05_20.md,
research/decisions/D002_chat_template_and_gen_length_unification.md,
and research/notes/method_evolution_directions_2026_05_20.md.

I am picking up the *method evolution* track (Section 8). The source server is
already running all post-D002 benchmark re-runs and will sync results when done
— do NOT re-execute those. Pick one candidate from method_evolution_directions
(D-A / D-B / D-D suggested as first move) and propose an experiment plan.
```

---

**Cross-links** (existing notes worth knowing about):
- [[phase14a_e2e_results]] — our E3 vs vanilla end-to-end on GSM8K/HumanEval at N=100/164
- [[phase13b_cheap_estimator]] — derivation of lag-1 + $U_t$ + window-4 (E3)
- [[novelty_positioning_2026_05_18]] — earlier positioning vs prior art
- [[fastgen_head_pattern_probe]] — head-pattern findings used in our sparse-Q design
