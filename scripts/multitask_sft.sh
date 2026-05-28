#!/usr/bin/env bash
set -e

MODEL_NAME="src/models/LatentSeeker"
MODEL_CKPT="outputs/pretrain/stage4"  # paraphrase-trained weights
CONFIG_PATH="configs/multitask_sft.yaml"
OUTPUT_DIR="outputs/multitask_sft"
RESUME_FROM=""  # e.g. "outputs/multitask_sft/checkpoint-100"
mkdir -p "$OUTPUT_DIR"

if [ ! -d "data/wiki/processed_wiki" ]; then
    echo "Preprocessing wiki dataset..."
    python src/dataset/preprocess_wiki.py \
        --input data/wiki/20220301.en.jsonl \
        --output data/wiki/processed_wiki
fi

if [ ! -d "data/gen_wiki/processed_gen_qa" ]; then
    if [ -f "data/gen_wiki/gen_qa.jsonl" ]; then
        echo "Preprocessing QA dataset..."
        python -m src.dataset.preprocess_qa \
            --input data/gen_wiki/gen_qa.jsonl \
            --output data/gen_wiki/processed_gen_qa \
            --max-turns 20
    else
        echo "QA data not found at data/gen_wiki/gen_qa.jsonl"
        echo "Run scripts/gen_qa.sh first to generate multi-turn QA data."
        exit 1
    fi
fi

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed main.py \
    --config_path "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --model_ckpt_path "$MODEL_CKPT" \
    ${RESUME_FROM:+--resume_from_checkpoint "$RESUME_FROM"}
