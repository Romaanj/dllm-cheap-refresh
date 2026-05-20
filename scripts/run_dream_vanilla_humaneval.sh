#!/bin/bash
# Dream-v0-Instruct-7B, vanilla baseline, HumanEval gen=512, chat ON.
source "$(dirname "$0")/_common.sh"
run_eval_dream dream_vanilla_humaneval_g512_chaton \
  "model_path=$MODEL_DREAM,gen_length=512,steps=512,block_length=32,cheap_e3=False,threshold=0.0,show_speed=True" \
  humaneval 0
