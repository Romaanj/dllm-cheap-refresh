#!/bin/bash
# Dream-v0-Instruct-7B, Ours (cheap E3 uniform K=256), HumanEval gen=512, chat ON.
source "$(dirname "$0")/_common.sh"
run_eval_dream dream_ours_humaneval_g512_chaton \
  "model_path=$MODEL_DREAM,gen_length=512,steps=512,block_length=32,cheap_e3=True,cheap_K=256,cheap_k_schedule=uniform,cheap_warmup=2,threshold=0.9,show_speed=True" \
  humaneval 0
