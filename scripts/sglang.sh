#!/bin/bash
# filepath: /data/yanggb/workspace/ReCall/launch_sglang.sh

export CUDA_VISIBLE_DEVICES=2,3
MODEL_NAME="qwen3b"
MODEL_PATH="/mnt/model/Qwen2.5-3B-Instruct"

python3 -m sglang.launch_server \
    --served-model-name ${MODEL_NAME} \
    --model-path ${MODEL_PATH} \
    --tp 2 \
    --context-length 20992 \
    --enable-metrics \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 80 \
    --trust-remote-code \
    --disable-overlap \
    --disable-radix-cach