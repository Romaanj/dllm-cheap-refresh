#!/bin/bash
# LLaDA-8B-Instruct, Fast-dLLM (dual cache + threshold parallel), HumanEval gen=512, chat ON.
source "$(dirname "$0")/_common.sh"
run_eval_llada llada_fast_dllm_humaneval_g512_chaton \
  "model_path=$MODEL_LLADA,gen_length=512,steps=512,block_length=32,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True" \
  humaneval 0
