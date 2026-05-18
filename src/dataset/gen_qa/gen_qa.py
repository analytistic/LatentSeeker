#!/usr/bin/env python3
"""Generate multi-turn QA training data from a JSONL corpus.

Iteratively generates one QA turn at a time with real token counting,
diverse question types, and Qwen3-compatible ``reasoning_content`` format.

Usage:
    python -m src.dataset.gen_qa.gen_qa \\
        --input data/wiki/wiki_15k.jsonl \\
        --output data/wiki/qa.jsonl \\
        --model deepseek-v4-flash \\
        --tokenizer-path src/models/LatentSeeker

    python -m src.dataset.gen_qa.gen_qa \\
        --input data/wiki.jsonl \\
        --output data/qa.jsonl \\
        --model /path/to/model \\
        --api-base http://localhost:8000/v1 \\
        --api-protocol openai \\
        --max-workers 16 \\
        --question-types "summary:2,detail:2,needle:1,multi_hop:1,follow_up:2,synthesis:1"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from datasets import load_dataset
from transformers import AutoTokenizer

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
    "synthesis": {
        "instruction": "Ask a question that combines information from both the document and the previous conversation turns to reach a new insight or synthesis.",
        "source": "both",
    },
}


def parse_type_spec(spec: str) -> list[str]:
    """Parse ``"summary:2,detail:1,follow_up:2"`` into a weighted list.

    Each item is ``<name>`` or ``<name>:<weight>``.
    Returns a list where each type appears ``weight`` times.
    """
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

ADDITIVE_PROMPT = """Document:
{document}
{history_section}
Now generate the next QA turn.

Question type: {type_instruction}

Output exactly in this format:
Question: <question>
Reasoning: <step-by-step reasoning>
Answer: <concise answer>"""


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


def parse_one_turn(text: str) -> dict[str, str] | None:
    m = TURN_RE.search(text)
    if not m:
        return None
    return {
        "question": m.group(1).strip(),
        "reasoning": m.group(2).strip(),
        "answer": m.group(3).strip(),
    }


def format_assistant(turn: dict[str, str]) -> str:
    return f"Reasoning: {turn['reasoning']}\n\nAnswer: {turn['answer']}"


def build_messages(
    doc_text: str, turns: list[dict[str, str]]
) -> list[dict[str, Any]] | None:
    """Build LatentSeeker messages with Qwen3 ``reasoning_content`` format."""
    if not turns:
        return None
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "longtext", "longtext": doc_text},
                {"type": "text", "text": turns[0]["question"]},
            ],
        }
    ]
    for i, t in enumerate(turns):
        msgs.append(
            {
                "role": "assistant",
                "reasoning_content": t["reasoning"],
                "content": [{"type": "text", "text": t["answer"]}],
            }
        )
        if i < len(turns) - 1:
            msgs.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": turns[i + 1]["question"]}],
                }
            )
    return msgs


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


# ── Document-level generation (additive, token-counted) ─────────────────

def process_doc(
    text: str,
    tokenizer: AutoTokenizer,
    api_base: str,
    api_key: str,
    protocol: Literal["anthropic", "openai"],
    model: str,
    budget_tokens: int,
    max_tokens_per_call: int,
    temperature: float,
    question_pool: list[str],
) -> tuple[list[dict[str, str]], int, bool, float] | None:
    """Generate turns additively, counting real tokens.
    Returns ``(turns, total_assistant_tokens, budget_exceeded, api_time_s)`` or ``None``.
    """
    if not text or len(text) < 50:
        return None

    turns: list[dict[str, str]] = []
    total_tokens = 0
    api_time_s = 0.0
    max_retries = 3
    budget_exceeded = False
    last_type: str | None = None
    while True:
        # 1. Sample question type by weight, avoid consecutive repeats
        qtype_name = sample_type(question_pool, last_type)
        last_type = qtype_name
        qtype = QUESTION_TYPE_DEFS[qtype_name]
        type_instruction = qtype["instruction"]

        # 2. Build prompt: document + conversation history (for context)
        history = format_history(turns)
        prompt = ADDITIVE_PROMPT.format(
            document=text,
            history_section=history or "",
            type_instruction=type_instruction,
        )

        # 3. Call API (with retries)
        content = None
        for attempt in range(max_retries):
            content, elapsed = call_api(
                api_base, api_key, protocol, model,
                SYSTEM_PROMPT, prompt,
                max_tokens_per_call, temperature,
            )
            api_time_s += elapsed
            if content:
                break
            print(f"    [retry {attempt + 1}/{max_retries}]", file=sys.stderr)

        if not content:
            break

        # 4. Parse single turn
        turn = parse_one_turn(content)
        if not turn:
            print(f"    [warn] unparseable turn, skipping", file=sys.stderr)
            continue

        # 5. Count total Q+A tokens via chat template (longtext is skipped)
        candidate_turns = turns + [turn]
        msgs = build_messages(text, candidate_turns)
        if msgs is None:
            continue
        candidate_total = len(
            tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        )
        # 6. If budget exceeded: keep this turn in messages but stop
        if candidate_total > budget_tokens:
            budget_exceeded = True
            turns.append(turn)
            break

        turns.append(turn)
        total_tokens = candidate_total
    return (turns, total_tokens, budget_exceeded, api_time_s) if turns else None


# ── Checkpoint ──────────────────────────────────────────────────────────

_STATE_SUFFIX = ".gen_qa_progress.json"


def _save_progress(path: Path, done: set[int]) -> None:
    path.write_text(json.dumps({"done": sorted(done)}))


def _load_progress(path: Path) -> set[int]:
    return set(json.loads(path.read_text()).get("done", [])) if path.exists() else set()


# ── Worker ──────────────────────────────────────────────────────────────

def _worker(args: tuple) -> tuple[int, tuple | None]:
    (
        idx,
        text,
        tokenizer,
        api_base,
        api_key,
        protocol,
        model,
        budget_tokens,
        max_tokens_per_call,
        temperature,
        question_pool,
    ) = args
    result = process_doc(
        text, tokenizer, api_base, api_key, protocol, model,
        budget_tokens, max_tokens_per_call, temperature,
        question_pool,
    )
    if result:
        return idx, result
    return idx, None


# ── CLI ─────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate multi-turn QA training data from JSONL"
    )
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
        help="Token budget for all Q&A messages combined via chat template (excluding document)",
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
    return p


def main() -> None:
    args = _build_parser().parse_args()
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    question_pool = parse_type_spec(args.question_types)

    # ── Tokenizer (for real token counting) ───────────────────────
    print(f"Loading tokenizer from {args.tokenizer_path} …")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"  vocab_size={tokenizer.vocab_size}")

    # ── Load input (Arrow-backed, lazy) ──────────────────────────
    dataset = load_dataset("json", data_files=args.input, split="train")
    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    total = len(dataset)
    state_path = Path(args.input).with_suffix(_STATE_SUFFIX)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume ───────────────────────────────────────────────────
    done = _load_progress(state_path) if args.resume else set()
    remaining = [i for i in range(total) if i not in done]

    if not remaining:
        print("All documents already processed.")
        return

    print(f"Total: {total}  Done: {len(done)}  Remaining: {len(remaining)}")
    print(f"Question types: {dict(Counter(question_pool))}")

    mode = "a" if args.resume and out_path.exists() else "w"
    completed = errors = 0
    start_time = time.time()

    def _eta(elapsed: float, done: int, total: int) -> str:
        if done == 0:
            return "--:--:--"
        avg = elapsed / done
        rem = avg * (total - done)
        h, r = divmod(int(rem), 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    with open(out_path, mode) as out:
        if args.max_workers <= 1:
            for idx in remaining:
                text = dataset[idx]["text"]
                result = process_doc(
                    text, tokenizer,
                    args.api_base, api_key, args.api_protocol, args.model,
                    args.max_qa_tokens, args.max_tokens_per_call,
                    args.temperature, question_pool,
                )
                turns_info = None
                if result:
                    turns, total_tokens, budget_exceeded, api_time_s = result
                    msgs = build_messages(text, turns)
                    if msgs:
                        num_fitting = len(turns) - 1 if budget_exceeded else len(turns)
                        turns_info = {"num_turns": num_fitting, "assistant_token_len": total_tokens}
                        out.write(
                            json.dumps(
                                {"messages": msgs, **turns_info},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        out.flush()
                        completed += 1
                    else:
                        errors += 1
                else:
                    errors += 1
                done.add(idx)
                _save_progress(state_path, done)
                n = turns_info["num_turns"] if turns_info else 0
                eta = _eta(time.time() - start_time, completed, len(remaining))
                print(
                    f"  [{completed}/{len(remaining)}]  doc {idx}  "
                    f"ok={completed}  err={errors}  "
                    f"turns={n}  api={api_time_s:.1f}s  eta={eta}"
                )
        else:
            worker_args = [
                (
                    idx,
                    dataset[idx]["text"],
                    tokenizer,
                    args.api_base,
                    api_key,
                    args.api_protocol,
                    args.model,
                    args.max_qa_tokens,
                    args.max_tokens_per_call,
                    args.temperature,
                    question_pool,
                )
                for idx in remaining
            ]
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {pool.submit(_worker, a): a[0] for a in worker_args}
                for future in as_completed(futures):
                    idx = futures[future]
                    result = None
                    try:
                        _, result = future.result()
                    except Exception as e:
                        print(f"  [error] doc {idx}: {e}", file=sys.stderr)
                    turns_info = None
                    if result:
                        turns, total_tokens, budget_exceeded, api_time_s = result
                        msgs = build_messages(dataset[idx]["text"], turns)
                        if msgs:
                            num_fitting = len(turns) - 1 if budget_exceeded else len(turns)
                            turns_info = {"num_turns": num_fitting, "assistant_token_len": total_tokens}
                            out.write(
                                json.dumps(
                                    {"messages": msgs, **turns_info},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            out.flush()
                            completed += 1
                        else:
                            errors += 1
                    else:
                        errors += 1
                    done.add(idx)
                    _save_progress(state_path, done)
                    n = turns_info["num_turns"] if turns_info else 0
                    eta = _eta(time.time() - start_time, completed, len(remaining))
                    print(
                        f"  [{completed}/{len(remaining)}]  doc {idx}  "
                        f"ok={completed}  err={errors}  "
                        f"turns={n}  api={api_time_s:.1f}s  eta={eta}"
                    )

    print(f"\nDone.  Successful: {completed}  Errors: {errors}")


if __name__ == "__main__":
    main()
