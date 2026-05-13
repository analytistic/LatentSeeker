#!/usr/bin/env bash
set -e

PYTHON=".venv/bin/python"
OUTPUT_DIR="outputs/debug"
RESUME_FROM=""
mkdir -p "$OUTPUT_DIR"

if [ ! -d "data/debug/processed_debug" ]; then
    echo "Preprocessing debug dataset..."
    $PYTHON src/dataset/preprocess_wiki.py \
        --input data/debug/debug.jsonl \
        --output data/debug/processed_debug
fi

$PYTHON main.py --config_path configs/debug.yaml \
    --output_dir "$OUTPUT_DIR" \
    --bf16 false \
    --use_cpu true \
    --deepspeed "" \
    ${RESUME_FROM:+--resume_from_checkpoint "$RESUME_FROM"}
