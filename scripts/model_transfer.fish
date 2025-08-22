#!/usr/bin/fish

# usage: fish model_transfer.fish [RUN_NAME] [TRAIN_STEP]

conda activate recall

if test (count $argv) -ne 1
    echo "Usage: fish model_transfer.fish [TRAIN_STEP]"
    echo "Example: fish model_transfer.fish 100"
    exit 1
end

set -x CUDA_VISIBLE_DEVICES 2,3
set -x MODEL_NAME Qwen2.5-7B-Instruct
set -x TRAIN_STEP $argv[1]

set -x MODEL_PATH /mnt/llmshared-ssd-hd/yangganbo/outputs/checkpoints/global_step_$TRAIN_STEP/actor
set -x TARGET_PATH /mnt/llmshared-ssd-hd/yangganbo/outputs/checkpoints/hf_model_$TRAIN_STEP

if not test -d $MODEL_PATH
    echo "Error: Model path $MODEL_PATH does not exist"
    exit 1
end

mkdir -p $TARGET_PATH

cp -r /mnt/llmshared-ssd-hd/lanhengzhi/output/train/$MODEL_NAME/model_info/* $MODEL_PATH
cp -r /mnt/llmshared-ssd-hd/lanhengzhi/output/train/$MODEL_NAME/model_info/* $TARGET_PATH

python model_merger.py merge \
    --backend fsdp \
    --local_dir $MODEL_PATH \
    --target_dir $TARGET_PATH