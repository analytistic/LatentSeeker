#!/usr/bin/env python3
"""Generate multi-turn longtext QA data with continuous dialogue.

Each conversation contains multiple rounds, where each round introduces
a new longtext document followed by a question and answer. Questions can
reference or build upon the previous conversation.

Input format (JSONL, one sample per line):
    {"documents": ["doc1 text...", "doc2 text...", ...]}

Output format:
    messages: [
        {role: user, content: [longtext(doc1), text(q1)]},
        {role: assistant, reasoning_content(r1), content(text(a1))},
        {role: user, content: [longtext(doc2), text(q2)]},
        {role: assistant, reasoning_content(r2), content(text(a2))},
        ...
    ]

Usage:
    python -m src.dataset.gen_qa.gen_qa_multi_turn \\
        --input data/multi_docs.jsonl \\
        --output data/qa_multi_turn.jsonl \\
        --max-docs 5 \\
        --max-qa-tokens 3000

    python -m src.dataset.gen_qa.gen_qa_multi_turn \\
        --input data/multi_docs.jsonl \\
        --output data/qa_multi_turn.jsonl \\
        --max-docs 3 \\
        --max-workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from datasets import load_dataset
from transformers import AutoTokenizer

from .base import (
    QUESTION_TYPE_DEFS,
    SYSTEM_PROMPT,
    add_common_args,
    call_api,
    load_progress,
    parse_one_turn,
    parse_type_spec,
    sample_type,
    save_progress,
)

# ── Prompts ─────────────────────────────────────────────────────────────

MULTI_TURN_PROMPT = """Document:
{document}
{history_section}
Now generate a question about the NEW document above.
The question can reference or build upon the previous conversation.

Question type: {type_instruction}

Output exactly in this format:
Question: <question>
Reasoning: <step-by-step reasoning>
Answer: <concise answer>"""


def format_history_with_docs(
    turns: list[dict[str, str]], doc_indices: list[int]
) -> str:
    """Format previous conversation rounds with document references."""
    if not turns:
        return ""
    lines = ["\nPrevious conversation:"]
    for i, (t, doc_idx) in enumerate(zip(turns, doc_indices)):
        lines.append(f"Round {i+1} (Document {doc_idx+1}):")
        lines.append(f"  Q{i+1}: {t['question']}")
        lines.append(f"  A{i+1}: {t['answer']}")
    return "\n".join(lines)


# ── Message building ────────────────────────────────────────────────────

def build_messages_multi_turn(
    docs: list[str], turns: list[dict[str, str]]
) -> list[dict[str, Any]] | None:
    """Build messages for multi-turn longtext conversations.

    Each round adds: user[longtext(doc_i), text(q_i)] + assistant[r_i, text(a_i)]
    """
    if not turns or not docs:
        return None
    msgs = []
    for i, (doc, turn) in enumerate(zip(docs, turns)):
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "longtext", "longtext": doc},
                    {"type": "text", "text": turn["question"]},
                ],
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "reasoning_content": turn["reasoning"],
                "content": [{"type": "text", "text": turn["answer"]}],
            }
        )
    return msgs


# ── Document-level generation ───────────────────────────────────────────

def process_doc_multi_turn(
    documents: list[str],
    tokenizer: AutoTokenizer,
    api_base: str,
    api_key: str,
    protocol: Literal["anthropic", "openai"],
    model: str,
    budget_tokens: int,
    max_tokens_per_call: int,
    temperature: float,
    question_pool: list[str],
    max_docs: int,
) -> tuple[list[dict[str, str]], list[int], int, bool, float] | None:
    """Generate multi-turn longtext conversation.

    Each round introduces a new document with a question.
    Returns ``(turns, doc_indices, total_tokens, budget_exceeded, api_time_s)``
    or ``None``.
    """
    if not documents:
        return None

    # Limit number of documents
    n_docs = min(max_docs, len(documents))
    selected_indices = list(range(n_docs))
    selected_docs = [documents[i] for i in selected_indices]

    turns: list[dict[str, str]] = []
    doc_indices_used: list[int] = []
    total_tokens = 0
    api_time_s = 0.0
    max_retries = 3
    budget_exceeded = False
    last_type: str | None = None

    for round_idx in range(n_docs):
        current_doc = selected_docs[round_idx]

        # Sample question type
        qtype_name = sample_type(question_pool, last_type)
        last_type = qtype_name
        qtype = QUESTION_TYPE_DEFS[qtype_name]
        type_instruction = qtype["instruction"]

        # Build prompt with history
        history = format_history_with_docs(turns, doc_indices_used)
        prompt = MULTI_TURN_PROMPT.format(
            document=current_doc,
            history_section=history or "",
            type_instruction=type_instruction,
        )

        # Call API (with retries)
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

        # Parse single turn
        turn = parse_one_turn(content)
        if not turn:
            print(f"    [warn] unparseable turn at round {round_idx}, skipping",
                  file=sys.stderr)
            continue

        # Count total Q+A tokens (longtext is skipped in token counting)
        candidate_turns = turns + [turn]
        candidate_docs = selected_docs[: len(candidate_turns)]
        msgs = build_messages_multi_turn(candidate_docs, candidate_turns)
        if msgs is None:
            continue
        candidate_total = len(
            tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                multi_turn_reasoning=True,
            )["input_ids"]
        )

        # If budget exceeded: keep this turn but stop
        if candidate_total > budget_tokens:
            budget_exceeded = True
            turns.append(turn)
            doc_indices_used.append(selected_indices[round_idx])
            break

        turns.append(turn)
        doc_indices_used.append(selected_indices[round_idx])
        total_tokens = candidate_total

    return (turns, doc_indices_used, total_tokens, budget_exceeded, api_time_s) if turns else None


# ── Worker ──────────────────────────────────────────────────────────────

def _worker(args: tuple) -> tuple[int, tuple | None]:
    (
        idx,
        documents,
        tokenizer,
        api_base,
        api_key,
        protocol,
        model,
        budget_tokens,
        max_tokens_per_call,
        temperature,
        question_pool,
        max_docs,
    ) = args
    result = process_doc_multi_turn(
        documents, tokenizer, api_base, api_key, protocol, model,
        budget_tokens, max_tokens_per_call, temperature,
        question_pool, max_docs,
    )
    if result:
        return idx, result
    return idx, None


# ── CLI ─────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate multi-turn longtext QA with continuous dialogue"
    )
    add_common_args(p)
    p.add_argument(
        "--max-docs",
        type=int,
        default=5,
        help="Maximum number of documents (rounds) per conversation",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    question_pool = parse_type_spec(args.question_types)

    print(f"Loading tokenizer from {args.tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"  vocab_size={tokenizer.vocab_size}")

    dataset = load_dataset("json", data_files=args.input, split="train")
    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    total = len(dataset)
    state_path = Path(args.input).with_suffix(_STATE_SUFFIX)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_progress(state_path) if args.resume else set()
    remaining = [i for i in range(total) if i not in done]

    if not remaining:
        print("All samples already processed.")
        return

    print(f"Total: {total}  Done: {len(done)}  Remaining: {len(remaining)}")
    print(f"Max docs per conversation: {args.max_docs}")

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
                documents = dataset[idx]["documents"]
                result = process_doc_multi_turn(
                    documents, tokenizer,
                    args.api_base, api_key, args.api_protocol, args.model,
                    args.max_qa_tokens, args.max_tokens_per_call,
                    args.temperature, question_pool, args.max_docs,
                )
                turns_info = None
                if result:
                    turns, doc_indices, total_tokens, budget_exceeded, api_time_s = result
                    used_docs = [documents[i] for i in doc_indices]
                    msgs = build_messages_multi_turn(used_docs, turns)
                    if msgs:
                        num_fitting = len(turns) - 1 if budget_exceeded else len(turns)
                        turns_info = {
                            "num_turns": num_fitting,
                            "num_rounds": len(turns),
                            "assistant_token_len": total_tokens,
                        }
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
                save_progress(state_path, done)
                n = turns_info["num_turns"] if turns_info else 0
                eta = _eta(time.time() - start_time, completed, len(remaining))
                print(
                    f"  [{completed}/{len(remaining)}]  sample {idx}  "
                    f"ok={completed}  err={errors}  "
                    f"rounds={n}  api={api_time_s:.1f}s  eta={eta}"
                )
        else:
            worker_args = [
                (
                    idx,
                    dataset[idx]["documents"],
                    tokenizer,
                    args.api_base,
                    api_key,
                    args.api_protocol,
                    args.model,
                    args.max_qa_tokens,
                    args.max_tokens_per_call,
                    args.temperature,
                    question_pool,
                    args.max_docs,
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
                        print(f"  [error] sample {idx}: {e}", file=sys.stderr)
                    turns_info = None
                    if result:
                        turns, doc_indices, total_tokens, budget_exceeded, api_time_s = result
                        documents = dataset[idx]["documents"]
                        used_docs = [documents[i] for i in doc_indices]
                        msgs = build_messages_multi_turn(used_docs, turns)
                        if msgs:
                            num_fitting = len(turns) - 1 if budget_exceeded else len(turns)
                            turns_info = {
                                "num_turns": num_fitting,
                                "num_rounds": len(turns),
                                "assistant_token_len": total_tokens,
                            }
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
                    save_progress(state_path, done)
                    n = turns_info["num_turns"] if turns_info else 0
                    eta = _eta(time.time() - start_time, completed, len(remaining))
                    print(
                        f"  [{completed}/{len(remaining)}]  sample {idx}  "
                        f"ok={completed}  err={errors}  "
                        f"rounds={n}  api={api_time_s:.1f}s  eta={eta}"
                    )

    print(f"\nDone.  Successful: {completed}  Errors: {errors}")


if __name__ == "__main__":
    main()
