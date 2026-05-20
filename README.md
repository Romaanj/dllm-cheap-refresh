# dllm-cheap-refresh

Research on **cheap-refresh inference acceleration for diffusion language models (dLLMs)**, comparing against published caching baselines (Fast-dLLM, Dynamic-dLLM, Elastic-Cache) on LLaDA-8B-Instruct and Dream-v0-Instruct-7B.

Our method: **E3** = lag-1 + remaining-unmask budget ($U_t$) + window-4, paired with sparse-Q forward and KV cache, composed with $\theta{=}0.9$ confidence-threshold parallel decoding.

## Quick start (new environment)

```bash
git clone https://github.com/Romaanj/dllm-cheap-refresh.git
cd dllm-cheap-refresh

# 1. baselines vendored
./scripts/clone_baselines.sh

# 2. python deps
pip install torch transformers accelerate lm-eval evaluate datasets einops tqdm

# 3. env
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# 4. resume context with Claude (or read manually)
claude "Read CLAUDE.md and research/notes/session_state_2026_05_20.md, then continue."
```

## Entry points for context

- [`research/notes/session_state_2026_05_20.md`](research/notes/session_state_2026_05_20.md) — full project state, baseline numbers, pending re-runs
- [`research/decisions/D002_chat_template_and_gen_length_unification.md`](research/decisions/D002_chat_template_and_gen_length_unification.md) — the most recent methodology decision (chat ON + gen=512 for HumanEval)
- [`research/INDEX.md`](research/INDEX.md) — full notes index

## Layout

```
methods/
  ours/                  # E3 cheap refresh: code + entry-point evals
    eval_llada.py        # lm-eval adapter for LLaDA (Ours + all LLaDA-side baselines branching by flags)
    eval_dream.py        # lm-eval adapter for Dream
    cheap_e3_inference_llada.py  # main method impl (lag-1 + U_t + window-4)
    phase14a_cheap_e2e.py        # cascade forward + A_l builder
    phase9_day3_composed.py      # transfer / pick utilities
    sparse_block.py              # sparse-Q forward kernel scaffolding
    gsm8k_attention_sink_drift_eval.py
    drift_refresh_attention.py
    scope_incremental_attention.py
    sparse_attention_prototype.py
    phase14a_dream.py, phase12_cascade_oracle_dream.py  # Dream-side
    postprocess_code.py, sanitize.py                     # HumanEval scorer
    generate.py          # Fast-dLLM-original generation (vendored)
    model/               # Fast-dLLM-original LLaDA model (vendored)
  fast_dllm/             # NVlabs/Fast-dLLM (cloned by clone_baselines.sh)
  dynamic_dllm/          # Dynamic-DLLM (cloned)
  elastic_cache/         # YuyangSunshine/Elastic-Cache (cloned)

scripts/                 # run_<model>_<method>_<task>.sh wrappers
  run_llada_vanilla_humaneval.sh
  run_llada_fast_dllm_humaneval.sh
  run_llada_dynamic_dllm_humaneval.sh
  run_llada_dynamic_dllm_humaneval_no_apd.sh
  run_llada_ours_humaneval.sh
  run_dream_vanilla_humaneval.sh
  run_dream_ours_humaneval.sh
  postprocess_humaneval.sh
  clone_baselines.sh
  _common.sh             # shared env + helper functions

research/                # notes, lit reviews, hypotheses, decisions
  INDEX.md
  notes/session_state_2026_05_20.md  # ← READ FIRST
  decisions/D002_chat_template_and_gen_length_unification.md
  ...

tables.tex               # LaTeX: main results + Dynamic-dLLM ablation
CLAUDE.md                # Claude Code workflow rules for this project
```

## Current re-run set (HumanEval, post-D002)

All LLaDA + Dream HumanEval baselines need re-run with chat template ON and gen_length=512. See [D002](research/decisions/D002_chat_template_and_gen_length_unification.md) for protocol details and `scripts/README.md` for parallelization.

## Source / attribution

Replaces a workspace that lived inside a fork of [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM). The vendored `methods/ours/generate.py` and `methods/ours/model/` are derived from Fast-dLLM. Baselines under `methods/{fast_dllm, dynamic_dllm, elastic_cache}/` are fresh clones from their respective origins (see `scripts/clone_baselines.sh`).
