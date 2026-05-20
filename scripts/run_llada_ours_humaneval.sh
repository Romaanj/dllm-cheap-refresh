#!/bin/bash
# LLaDA-8B-Instruct, Ours (cheap E3 + threshold parallel), K=384 stepped, HumanEval gen=512, chat ON.
source "$(dirname "$0")/_common.sh"
run_eval_llada llada_ours_humaneval_g512_chaton \
  "model_path=$MODEL_LLADA,gen_length=512,steps=512,block_length=32,cheap_e3=True,cheap_K=384,cheap_k_schedule=stepped,cheap_warmup=2,threshold=0.9,show_speed=True" \
  humaneval 0
