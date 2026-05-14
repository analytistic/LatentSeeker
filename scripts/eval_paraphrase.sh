#!/usr/bin/env bash
set -e

DATASET="debug"
MODEL_PATH="outputs/pretrain/stage1/checkpoint-500"
DATA_DIR="data/$DATASET"
DATA_PATH="$DATA_DIR/processed_$DATASET"

if [ ! -d "$DATA_PATH" ]; then
    echo "Preprocessing $DATASET dataset..."
    .venv/bin/python src/dataset/preprocess_wiki.py \
        --input "$DATA_DIR/${DATASET}.jsonl" \
        --output "$DATA_PATH"
fi

python -m src.evaluation.eval_paraphrase \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --compress_ratio 1 2 4 \
    --max_new_tokens 512
