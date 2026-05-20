#!/bin/bash
# Score HumanEval samples with sanitize + hf_evaluate code_eval.
# Usage: ./scripts/postprocess_humaneval.sh <path/to/samples_humaneval_*.jsonl>
set -euo pipefail
SAMPLES="$1"
[ -z "$SAMPLES" ] && { echo "usage: $0 <samples.jsonl>"; exit 1; }
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# /dev/shm cleanup helps avoid postprocess EOFError from KMP_REGISTERED_LIB leaks.
rm -f /dev/shm/__KMP_REGISTERED_LIB_* 2>/dev/null || true
cd "$REPO_ROOT/methods/ours"
HF_ALLOW_CODE_EVAL=1 python postprocess_code.py "$SAMPLES"
