#!/usr/bin/env bash
set -e

# Generate multi-turn longtext QA data
# Usage: bash scripts/gen_qa_multi_turn.sh [--max-samples N]

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-your-key-here}"

INPUT="${INPUT:-data/wiki/wiki_15k.jsonl}"
OUTPUT="${OUTPUT:-data/wiki/qa_multi_turn.jsonl}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_SAMPLES="$1"

CMD="python3 -m src.dataset.gen_qa.gen_qa_multi_turn \
    --input \"$INPUT\" \
    --output \"$OUTPUT\" \
    --model deepseek-v4-flash \
    --api-base https://api.deepseek.com/anthropic \
    --api-protocol anthropic \
    --tokenizer-path src/models/LatentSeeker \
    --max-qa-tokens 1500 \
    --max-tokens-per-call 4096 \
    --temperature 0.7 \
    --max-docs 8 \
    --max-group-size 3 \
    --max-group-query-num 3 \
    --max-workers $MAX_WORKERS"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

echo "Running: $CMD"
eval "$CMD"
