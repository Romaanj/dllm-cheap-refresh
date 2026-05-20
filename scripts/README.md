# scripts/

Launch wrappers for HumanEval (gen=512, chat ON) — the post-D002 protocol.

## Layout

- `clone_baselines.sh` — fetches Fast-dLLM, Dynamic-DLLM, Elastic-Cache fresh from origin
- `run_<model>_<method>_humaneval.sh` — one wrapper per (model × method) for HumanEval
- `run_<model>_<method>_gsm8k.sh` — same for GSM8K
- `postprocess_humaneval.sh <samples_jsonl>` — sanitize + code_eval scorer (HumanEval only)

## Common env vars

```bash
export OUT_ROOT=/path/to/results        # results dir
export MODEL_LLADA=GSAI-ML/LLaDA-8B-Instruct
export MODEL_DREAM=Dream-org/Dream-v0-Instruct-7B
export GPU_ID=0                          # which GPU
```

If unset, scripts default to `OUT_ROOT=./results`, `GPU_ID=0`.

## Required setup before first run

```bash
./scripts/clone_baselines.sh           # ~3 GB clone, vendored snapshot
pip install -r requirements.txt        # lm-eval-harness, accelerate, transformers, evaluate
export HF_ALLOW_CODE_EVAL=1            # for HumanEval scoring
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

## HumanEval re-run set (D002, gen=512, chat ON)

Run sequentially per GPU (each script ~25-75 min):

```bash
# GPU 0
GPU_ID=0 ./scripts/run_llada_vanilla_humaneval.sh
GPU_ID=0 ./scripts/run_llada_fast_dllm_humaneval.sh
GPU_ID=0 ./scripts/run_llada_ours_humaneval.sh

# GPU 1
GPU_ID=1 ./scripts/run_llada_dynamic_dllm_humaneval.sh
GPU_ID=1 ./scripts/run_llada_dynamic_dllm_humaneval_no_apd.sh
GPU_ID=1 ./scripts/run_dream_vanilla_humaneval.sh
GPU_ID=1 ./scripts/run_dream_ours_humaneval.sh
```

After each run completes, score HumanEval with:

```bash
./scripts/postprocess_humaneval.sh $OUT_ROOT/<RUN_NAME>/*/samples_humaneval_*.jsonl
```

Elastic-Cache HumanEval has its own setup inside the vendored Elastic-Cache/ dir — use that dir's `eval_humaneval.sh` (model_path adjusted to `GSAI-ML/LLaDA-8B-Instruct`).
