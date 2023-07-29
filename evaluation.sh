#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

python3 github/lm-evaluation-harness/main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$2,dtype=float16,max_length=2048 \
  --tasks $3 \
  --batch_size 2 \
  --num_fewshot 0 \
  --device cuda:0 \
  --no_cache
