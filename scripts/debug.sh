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

if [ ! -d "data/debug/processed_qa" ]; then
    if [ -f "data/debug/qa.jsonl" ]; then
        echo "Preprocessing debug QA dataset..."
        $PYTHON -m src.dataset.preprocess_qa \
            --input data/debug/qa.jsonl \
            --output data/debug/processed_qa \
            --max-turns 4
    else
        echo "QA data not found at data/debug/qa.jsonl"
        echo "Run scripts/gen_qa_debug.sh first to generate QA data."
        exit 1
    fi
fi

$PYTHON main.py --config_path configs/debug.yaml \
    --output_dir "$OUTPUT_DIR" \
    --bf16 false \
    --use_cpu true \
    --deepspeed "" \
    ${RESUME_FROM:+--resume_from_checkpoint "$RESUME_FROM"}
