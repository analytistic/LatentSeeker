"""Shared utilities for QA generation scripts.

Extracted from gen_qa.py to avoid duplication across gen_qa_multi_text.py
and gen_qa_multi_turn.py.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal


# ── Question types ──────────────────────────────────────────────────────

QUESTION_TYPE_DEFS: dict[str, dict[str, str]] = {
    "summary": {
        "instruction": "Ask a question that requires summarizing the main themes, key points, or overall content of the document. The answer should synthesize information from across the document.",
        "source": "longtext",
    },
    "detail": {
        "instruction": "Ask a question about a specific fact, definition, number, name, or claim explicitly mentioned in the document.",
        "source": "longtext",
    },
    "needle": {
        "instruction": "Ask a question that requires finding a specific piece of information hidden in the document — such as a particular number, date, name, or statement. This should test the ability to locate precise information in long text.",
        "source": "longtext",
    },
    "multi_hop": {
        "instruction": "Ask a question that requires combining information from two or more separate parts of the document to arrive at the answer.",
        "source": "longtext",
    },
    "comparison": {
        "instruction": "Ask a question that compares or contrasts different concepts, entities, viewpoints, or pieces of information mentioned in the document.",
        "source": "longtext",
    },
    "temporal": {
        "instruction": "Ask a question about the sequence, chronology, order of events, or causal relationships described in the document.",
        "source": "longtext",
    },
    "follow_up": {
        "instruction": "Ask a follow-up question that builds on the previous Q&A turn. This could ask for elaboration, clarification, a deeper dive, or an implication of the previous answer.",
        "source": "history",
    },
    "evolve": {
        "instruction": "Based on the previous Q&A turn, rewrite the question to be more challenging. Increase reasoning depth, add constraints, combine multiple concepts, or introduce edge cases. Then answer the new question.",
        "source": "history",
    },
    "synthesis": {
        "instruction": "Ask a question that combines information from both the document and the previous conversation turns to reach a new insight or synthesis.",
        "source": "both",
    },
    "math_reasoning": {
        "instruction": "Ask a math or quantitative reasoning question based on numerical data, statistics, or quantitative information mentioned in the document. The question should require multi-step reasoning (calculations, comparisons, percentages, ratios, etc.) rather than a simple lookup.",
        "source": "longtext",
    },
}


def parse_type_spec(spec: str) -> list[str]:
    """Parse ``"summary:2,detail:1,follow_up:2"`` into a weighted list."""
    pool: list[str] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        name = parts[0].strip()
        weight = int(parts[1].strip()) if len(parts) > 1 else 1
        if name in QUESTION_TYPE_DEFS:
            pool.extend([name] * weight)
    if not pool:
        pool = list(QUESTION_TYPE_DEFS.keys())
    return pool


def sample_type(pool: list[str], last: str | None = None) -> str:
    """Sample a question type from the weighted pool, avoiding consecutive repeats."""
    if len(pool) == 1:
        return pool[0]
    while True:
        t = random.choice(pool)
        if t != last:
            return t


# ── Prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a data generation assistant. Generate high-quality multi-turn "
    "QA training data based on the provided document. Each turn must include "
    "step-by-step reasoning followed by a concise answer."
)


def format_history(turns: list[dict[str, str]]) -> str:
    if not turns:
        return ""
    lines = ["\nPrevious conversation:"]
    for i, t in enumerate(turns):
        lines.append(f"Q{i+1}: {t['question']}")
        lines.append(f"A{i+1}: {t['answer']}")
    return "\n".join(lines)


# ── Single-turn parsing ─────────────────────────────────────────────────

TURN_RE = re.compile(
    r"Question:\s*(.*?)\s*\n\s*Reasoning:\s*(.*?)\s*\n\s*Answer:\s*(.*)",
    re.DOTALL,
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_one_turn(text: str) -> dict[str, str] | None:
    cleaned = _THINK_RE.sub("", text).strip()
    m = TURN_RE.search(cleaned)
    if not m:
        return None
    return {
        "question": m.group(1).strip(),
        "reasoning": m.group(2).strip(),
        "answer": m.group(3).strip(),
    }


def format_assistant(turn: dict[str, str]) -> str:
    return f"Reasoning: {turn['reasoning']}\n\nAnswer: {turn['answer']}"


# ── HTTP API ────────────────────────────────────────────────────────────

def call_api(
    base: str,
    key: str,
    protocol: Literal["anthropic", "openai"],
    model: str,
    system: str,
    prompt: str,
    max_tokens: int,
    temp: float,
) -> tuple[str, float] | tuple[None, float]:
    """Returns ``(text, elapsed_seconds)`` or ``(None, elapsed)``."""
    t0 = time.perf_counter()
    if protocol == "anthropic":
        url = base.rstrip("/") + "/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temp,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        url = base.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temp,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  [error] HTTP {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
        return None, time.perf_counter() - t0
    except Exception as e:
        print(f"  [error] {e}", file=sys.stderr)
        return None, time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    try:
        if protocol == "anthropic":
            for b in result.get("content", []):
                if b.get("type") == "text":
                    return b["text"], elapsed
        else:
            return result["choices"][0]["message"]["content"], elapsed
    except (KeyError, IndexError, TypeError):
        print(f"  [error] bad response: {json.dumps(result)[:150]}", file=sys.stderr)
        return None, elapsed


# ── Checkpoint ──────────────────────────────────────────────────────────

_STATE_SUFFIX = ".gen_qa_progress.json"


def save_progress(path: Path, done: set[int]) -> None:
    path.write_text(json.dumps({"done": sorted(done)}))


def load_progress(path: Path) -> set[int]:
    return set(json.loads(path.read_text()).get("done", [])) if path.exists() else set()


# ── Shared CLI arguments ────────────────────────────────────────────────

def add_common_args(p) -> None:
    """Add common CLI arguments to an ArgumentParser."""
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--tokenizer-path", default="src/models/LatentSeeker")
    p.add_argument("--api-base", default="https://api.deepseek.com/anthropic")
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-protocol", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument(
        "--max-qa-tokens",
        type=int,
        default=1500,
        help="Token budget for all Q&A messages combined via chat template (excluding longtext)",
    )
    p.add_argument("--max-tokens-per-call", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument(
        "--question-types",
        default="summary:2,detail:2,needle:1,multi_hop:1,comparison:1,temporal:1,follow_up:2,synthesis:1",
        help="Colon-separated type:weight pairs, comma-separated",
    )
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--resume", action="store_true")


def make_worker_args(
    remaining: list[int],
    dataset,
    tokenizer,
    api_base,
    api_key,
    protocol,
    model,
    budget_tokens,
    max_tokens_per_call,
    temperature,
    question_pool,
    extra_kwargs: dict | None = None,
) -> list[tuple]:
    """Build argument tuples for worker functions."""
    base_args = [
        idx,
        dataset[idx],
        tokenizer,
        api_base,
        api_key,
        protocol,
        model,
        budget_tokens,
        max_tokens_per_call,
        temperature,
        question_pool,
    ]
    if extra_kwargs:
        base_args.append(extra_kwargs)
    return [tuple(a) for a in base_args]
