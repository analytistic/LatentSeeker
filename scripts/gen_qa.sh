#!/usr/bin/env bash
set -e

# Generate multi-turn QA training data from wiki 15k chunks
# Usage: bash scripts/gen_qa.sh [--max-samples N]

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# For a real run, set this in your env instead of here:
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-879b11c1900641a0adf028f1572ad731}"

INPUT="${INPUT:-data/wiki/wiki_15k.jsonl}"
OUTPUT="${OUTPUT:-data/wiki/processed_qa}"
MAX_WORKERS="${MAX_WORKERS:-1}"
MAX_SAMPLES="$1"

CMD="python3 -m src.dataset.gen_qa.gen_qa \
    --input \"$INPUT\" \
    --output \"$OUTPUT\" \
    --model deepseek-v4-flash \
    --api-base https://api.deepseek.com/anthropic \
    --api-protocol anthropic \
    --max-qa-tokens 1500 \
    --max-workers $MAX_WORKERS"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

echo "Running: $CMD"
eval "$CMD"
