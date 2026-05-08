#!/usr/bin/env bash
set -e

OUTPUT_DIR="outputs/debug"
mkdir -p "$OUTPUT_DIR"

if [ ! -d "data/debug/processed_debug" ]; then
    echo "Preprocessing debug dataset..."
    python src/dataset/preprocess_wiki.py \
        --input data/debug/debug.jsonl \
        --output data/debug/processed_debug
fi

ASCEND_RT_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 main.py --config_path configs/debug.yaml \
    --output_dir "$OUTPUT_DIR" \
    --use_cpu false \
    --deepspeed deepspeed/zero1.json
