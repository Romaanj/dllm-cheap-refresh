#!/bin/bash
# LLaDA-8B-Instruct, vanilla baseline (no cache, no parallel), HumanEval gen=512, chat ON (post-D002).
source "$(dirname "$0")/_common.sh"
run_eval_llada llada_vanilla_humaneval_g512_chaton \
  "model_path=$MODEL_LLADA,gen_length=512,steps=512,block_length=32,show_speed=True" \
  humaneval 0
