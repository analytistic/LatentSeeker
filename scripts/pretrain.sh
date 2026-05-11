#!/usr/bin/env bash
set -e

OUTPUT_DIR="outputs/pretrain/stage1"
mkdir -p "$OUTPUT_DIR"

if [ ! -d "data/wiki/processed_wiki" ]; then
    echo "Preprocessing wiki dataset..."
    python src/dataset/preprocess_wiki.py \
        --input data/wiki/wiki.jsonl \
        --output data/wiki/processed_wiki
fi

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed main.py \
    --config_path configs/pretrain_stage1.yaml \
    --output_dir "$OUTPUT_DIR" \
