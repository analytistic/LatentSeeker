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

python main.py --config_path configs/debug.yaml \
    --output_dir "$OUTPUT_DIR" \
    --bf16 false \
    --use_cpu true \
    --deepspeed ""
