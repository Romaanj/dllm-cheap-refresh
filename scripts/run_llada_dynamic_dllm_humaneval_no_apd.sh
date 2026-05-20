#!/bin/bash
# LLaDA-8B-Instruct, Dynamic-dLLM DCU only (APD disabled, pd_mode=0), HumanEval gen=512, chat ON.
# Used for the Dynamic-dLLM component-ablation table.
source "$(dirname "$0")/_common.sh"
run_eval_llada llada_dynamic_dllm_humaneval_g512_B64_no_apd_chaton \
  "model_path=$MODEL_LLADA,gen_length=512,steps=512,block_length=32,prompt_interval_steps=104,gen_interval_steps=8,use_cache=True,select_from=x,window_size=64,layer_budget=64,pd_mode=0,pd_threshold=0.9,alpha=0.001,beta=0.0008,device=cuda" \
  humaneval 0
