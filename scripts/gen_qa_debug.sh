#!/usr/bin/env bash
set -e

# Debug: generate multi-turn QA from debug.jsonl using DeepSeek API
# Usage: bash scripts/gen_qa_debug.sh

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-879b11c1900641a0adf028f1572ad731}"

python3 -m src.dataset.gen_qa.gen_qa \
    --input data/debug/debug.jsonl \
    --output data/debug/processed_qa \
    --model deepseek-v4-flash \
    --api-base https://api.deepseek.com/anthropic \
    --api-protocol anthropic \
    --max-samples 2 \
    --max-qa-tokens 1500 \
    --max-workers 4 \
    --temperature 0.7
