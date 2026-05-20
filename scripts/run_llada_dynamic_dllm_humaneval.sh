#!/bin/bash
# LLaDA-8B-Instruct, Dynamic-dLLM full (DCU + APD), paper config B=64 (auto-tune gen/8), HumanEval gen=512, chat ON.
# REQUIRES vendored Dynamic-DLLM cache/generate modules at methods/dynamic_dllm/.
# This wrapper calls eval_llada.py which expects llada_dist with pd_mode=2 (Dynamic-DLLM branch).
source "$(dirname "$0")/_common.sh"
run_eval_llada llada_dynamic_dllm_humaneval_g512_B64_chaton \
  "model_path=$MODEL_LLADA,gen_length=512,steps=512,block_length=32,prompt_interval_steps=104,gen_interval_steps=8,use_cache=True,select_from=x,window_size=64,layer_budget=64,pd_mode=2,pd_threshold=0.9,alpha=0.001,beta=0.0008,device=cuda" \
  humaneval 0
