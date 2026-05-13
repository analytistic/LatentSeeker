#!/usr/bin/env bash
set -e

DATASET="squad"
MODEL_PATH="outputs/pretrain/stage1/checkpoint-500"
SPLIT="validation"
OUTPUT_DIR="outputs/eval"
mkdir -p "$OUTPUT_DIR"

PROCESSED_DIR="data/$DATASET/processed"
if [ ! -d "$PROCESSED_DIR" ]; then
    echo "Preprocessing $DATASET..."
    python -m "src.dataset.preprocess_${DATASET}" \
        --input "data/$DATASET" \
        --output "data/$DATASET/processed"
fi

python -m src.evaluation.eval_qa \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --data_path "data/$DATASET/processed" \
    --split "$SPLIT" \
    --output "$OUTPUT_DIR/${DATASET}_${SPLIT}_preds.jsonl" \
    --compress_ratio 1 2 4 \
    --max_new_tokens 128
