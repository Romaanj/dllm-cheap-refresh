# Sourced by all run_*.sh wrappers. Sets defaults + helpers.
set -euo pipefail

: "${GPU_ID:=0}"
: "${OUT_ROOT:=./results}"
: "${MODEL_LLADA:=GSAI-ML/LLaDA-8B-Instruct}"
: "${MODEL_DREAM:=Dream-org/Dream-v0-Instruct-7B}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OURS_DIR="$REPO_ROOT/methods/ours"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

run_eval_llada() {
    local out_name="$1"; shift
    local model_args="$1"; shift
    local task="$1"; shift          # gsm8k or humaneval
    local fewshot="$1"; shift       # 5 or 0
    local out_dir="$OUT_ROOT/$out_name"
    local log_file="${out_dir}.log"
    mkdir -p "$out_dir"
    echo "[run] $out_name (GPU $GPU_ID, task=$task, fewshot=$fewshot)"
    echo "[run] log: $log_file"
    cd "$OURS_DIR"
    python eval_llada.py \
        --tasks "$task" --num_fewshot "$fewshot" --batch_size 1 --confirm_run_unsafe_code \
        --model llada_dist \
        --model_args "$model_args" \
        --output_path "$out_dir" --log_samples \
        2>&1 | tee "$log_file"
    cd "$REPO_ROOT"
}

run_eval_dream() {
    local out_name="$1"; shift
    local model_args="$1"; shift
    local task="$1"; shift
    local fewshot="$1"; shift
    local out_dir="$OUT_ROOT/$out_name"
    local log_file="${out_dir}.log"
    mkdir -p "$out_dir"
    echo "[run] $out_name (GPU $GPU_ID, task=$task, fewshot=$fewshot)"
    echo "[run] log: $log_file"
    cd "$OURS_DIR"
    accelerate launch --num_processes 1 eval_dream.py \
        --tasks "$task" --num_fewshot "$fewshot" --batch_size 1 --confirm_run_unsafe_code \
        --model dream_dist \
        --model_args "$model_args" \
        --output_path "$out_dir" --log_samples \
        2>&1 | tee "$log_file"
    cd "$REPO_ROOT"
}
