#!/bin/bash
# filepath: /data/yanggb/workspace/ReCall/launch_sglang.sh

MODEL_NAME="rollout3"
MODEL_PATH="/mnt/llmshared-ssd-hd/yangganbo/outputs/checkpoints/hf_model_206"

python3 -m sglang.launch_server \
    --served-model-name ${MODEL_NAME} \
    --model-path ${MODEL_PATH} \
    --tp 2 \
    --context-length 8192 \
    --enable-metrics \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 80 \
    --trust-remote-code \
    --disable-overlap \
    --disable-radix-cache