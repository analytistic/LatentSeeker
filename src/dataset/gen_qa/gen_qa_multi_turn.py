#!/usr/bin/env python3
"""Generate multi-turn longtext QA with group-based document streaming.

Documents are read one-by-one from the input stream and accumulated into
groups of variable size. Each group triggers multiple Q&A turns about
the full accumulated document set.

Supports parallel generation across conversations via ``--max-workers``.

Usage:
    python -m src.dataset.gen_qa.gen_qa_multi_turn \\
        --input data/docs.jsonl \\
        --output data/qa_multi_turn.jsonl \\
        --max-docs 8 \\
        --max-group-size 3 \\
        --max-group-query-num 3 \\
        --max-qa-tokens 3000
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Generator, Literal

from transformers import AutoTokenizer

from .base import (
    QUESTION_TYPE_DEFS,
    SYSTEM_PROMPT,
    _STATE_SUFFIX,
    add_common_args,
    call_api,
    parse_one_turn,
    parse_type_spec,
    sample_type,
)

# ── Prompts ─────────────────────────────────────────────────────────────

MULTI_TURN_PROMPT = """{conversation}
Now generate the next question about the documents above.

When referencing a document or part of it, be specific about the
position (e.g. "the first document", "Document 2") rather than
using vague references like "the above".

Question type: {type_instruction}

Output exactly in this format:
Question: <question>
Reasoning: <step-by-step reasoning>
Answer: <concise answer>"""


def format_conversation(
    groups: list[list[str]],
    turns: list[tuple[int, dict[str, str]]],
) -> str:
    """Format conversation with docs and Q&A grouped by batch."""
    blocks = []
    for g_idx, group in enumerate(groups):
        block = f"Group {g_idx + 1}:\n"
        for doc in group:
            block += f"\n{doc}\n---\n"
        for i, (gi, t) in enumerate(turns):
            if gi == g_idx:
                block += f"\nQ{i + 1}: {t['question']}\nR{i + 1}: {t['reasoning']}\nA{i + 1}: {t['answer']}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


# ── Message building ────────────────────────────────────────────────────

def build_messages(
    groups: list[list[str]],
    turns: list[tuple[int, dict[str, str]]],
) -> list[dict[str, Any]] | None:
    """Build messages with group-aware longtext placement.

    - First turn of a group: ``user [longtext(doc1), …, longtext(docN), text(q)]``
    - Subsequent turns within same group: ``user [text(q)]``
    - Assistant always: ``[text(answer)]`` with ``reasoning_content``.
    """
    if not turns or not groups:
        return None

    group_turn_count: dict[int, int] = {}
    msgs = []

    for group_idx, turn in turns:
        count = group_turn_count.get(group_idx, 0)

        if count == 0:
            group_docs = groups[group_idx]
            content: list[dict[str, str]] = [
                {"type": "longtext", "longtext": d} for d in group_docs
            ]
            content.append({"type": "text", "text": turn["question"]})
        else:
            content = [{"type": "text", "text": turn["question"]}]

        group_turn_count[group_idx] = count + 1

        msgs.append({"role": "user", "content": content})
        msgs.append(
            {
                "role": "assistant",
                "reasoning_content": turn["reasoning"],
                "content": [{"type": "text", "text": turn["answer"]}],
            }
        )

    return msgs


# ── Streaming input ────────────────────────────────────────────────────

def _iter_docs(path: str, start_line: int = 0) -> Generator[str, None, None]:
    """Stream JSONL lines, each ``{"text": "..."}``, optionally skipping."""
    with open(path) as f:
        for idx, line in enumerate(f):
            if idx < start_line:
                continue
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)["text"]


def _batch_docs(
    doc_iter: Generator[str, None, None], batch_size: int
) -> Generator[list[str], None, None]:
    """Read ``batch_size`` docs from iterator, yield one batch at a time."""
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(doc_iter))
            except StopIteration:
                break
        if not batch:
            return
        yield batch  # may be smaller than batch_size at end of file


# ── Conversation generation ────────────────────────────────────────────

def process_conversation(
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
    *,
    max_docs: int,
    max_group_size: int,
    max_group_query_num: int,
) -> tuple[list[dict[str, Any]], int, int, int, float] | None:
    """Generate one multi-turn conversation from a fixed list of documents.

    Returns ``(messages, num_turns, num_docs, total_tokens, api_time_s)``
    or ``None`` on failure.
    """
    if not documents:
        return None

    groups: list[list[str]] = []
    all_turns: list[tuple[int, dict[str, str]]] = []
    total_tokens = 0
    api_time_s = 0.0
    max_retries = 3
    budget_exceeded = False
    last_type: str | None = None
    pos = 0

    while sum(len(g) for g in groups) < max_docs and pos < len(documents):
        # ── 1. Sample group size ───────────────────────────────────
        remaining_docs = max_docs - sum(len(g) for g in groups)
        remaining_input = len(documents) - pos
        gs = random.randint(1, min(max_group_size, remaining_docs, remaining_input))

        # ── 2. Take ``gs`` docs from the input list ─────────────────
        group_docs = documents[pos:pos + gs]
        pos += gs

        group_idx = len(groups)
        groups.append(group_docs)
        group_has_turns = False

        # ── 3. Generate Q&A turns for this group ───────────────────
        n_queries = random.randint(1, max_group_query_num)
        for _ in range(n_queries):
            prompt = MULTI_TURN_PROMPT.format(
                conversation=format_conversation(groups, all_turns),
                type_instruction=QUESTION_TYPE_DEFS[
                    sample_type(question_pool, last_type)
                ]["instruction"],
            )

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

            turn = parse_one_turn(content)
            if not turn:
                print(f"    [warn] unparseable turn, skipping", file=sys.stderr)
                continue

            candidate_turns = all_turns + [(group_idx, turn)]
            msgs = build_messages(groups, candidate_turns)
            if msgs is None:
                continue
            candidate_total = len(
                tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False,
                    multi_turn_reasoning=True,
                )["input_ids"]
            )

            if candidate_total > budget_tokens:
                budget_exceeded = True
                all_turns.append((group_idx, turn))
                break

            all_turns.append((group_idx, turn))
            total_tokens = candidate_total
            group_has_turns = True

        if not group_has_turns:
            break
        if budget_exceeded:
            break

    if not all_turns:
        return None

    messages = build_messages(groups, all_turns)
    return messages, len(all_turns), sum(len(g) for g in groups), total_tokens, api_time_s


# ── Multi-thread helpers ───────────────────────────────────────────────

def _worker_task(
    batch_idx: int,
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
    max_group_size: int,
    max_group_query_num: int,
) -> tuple[int, dict | None]:
    """Wrapper around process_conversation for thread pool."""
    result = process_conversation(
        documents, tokenizer, api_base, api_key, protocol, model,
        budget_tokens, max_tokens_per_call, temperature, question_pool,
        max_docs=max_docs,
        max_group_size=max_group_size,
        max_group_query_num=max_group_query_num,
    )
    if result:
        messages, num_turns, num_docs, total_tokens, api_time_s = result
        return batch_idx, {
            "messages": messages,
            "num_turns": num_turns,
            "num_docs": num_docs,
            "assistant_token_len": total_tokens,
        }
    return batch_idx, None


# ── CLI ─────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate multi-turn longtext QA with group-based doc streaming"
    )
    add_common_args(p)
    p.add_argument("--max-docs", type=int, default=8,
                    help="Max total documents per conversation")
    p.add_argument("--max-group-size", type=int, default=3,
                    help="Max documents per group (random 1..N)")
    p.add_argument("--max-group-query-num", type=int, default=3,
                    help="Max Q&A turns per group (random 1..N)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    question_pool = parse_type_spec(args.question_types)

    print(f"Loading tokenizer from {args.tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"  vocab_size={tokenizer.vocab_size}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.input).with_suffix(_STATE_SUFFIX)

    # ── Resume state ──────────────────────────────────────────────
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text())
        samples_done = state.get("samples_done", 0)
        lines_read = state.get("lines_read", 0)
        print(f"Resuming: {samples_done} samples done, {lines_read} lines consumed")
    else:
        samples_done = 0
        lines_read = 0

    doc_iter = _iter_docs(args.input, start_line=lines_read)

    print(f"max_docs={args.max_docs}  max_group_size={args.max_group_size}  "
          f"max_group_query_num={args.max_group_query_num}  "
          f"budget_qa_tokens={args.max_qa_tokens}")

    mode = "a" if args.resume and out_path.exists() else "w"
    completed = errors = 0
    start_time = time.time()
    lock = threading.Lock()

    if args.max_workers <= 1:
        # ── Single-thread: read batch → process → write ──────────────
        with open(out_path, mode) as out:
            for batch in _batch_docs(doc_iter, args.max_docs):
                if args.max_samples and samples_done >= args.max_samples:
                    break
                lines_read += len(batch)

                result = process_conversation(
                    batch, tokenizer,
                    args.api_base, api_key, args.api_protocol, args.model,
                    args.max_qa_tokens, args.max_tokens_per_call,
                    args.temperature, question_pool,
                    max_docs=args.max_docs,
                    max_group_size=args.max_group_size,
                    max_group_query_num=args.max_group_query_num,
                )
                if result:
                    messages, num_turns, num_docs, total_tokens, api_time_s = result
                    out.write(
                        json.dumps({
                            "messages": messages,
                            "num_turns": num_turns,
                            "num_docs": num_docs,
                            "assistant_token_len": total_tokens,
                        }, ensure_ascii=False) + "\n"
                    )
                    out.flush()
                    completed += 1
                    samples_done += 1
                else:
                    errors += 1

                state_path.write_text(
                    json.dumps({"samples_done": samples_done, "lines_read": lines_read})
                )
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"  [{completed}]  turns={num_turns if result else '?'}  "
                    f"docs={num_docs if result else '?'}  "
                    f"tokens={total_tokens if result else '?'}  "
                    f"api={api_time_s if result else 0:.1f}s  {rate:.2f} conv/s"
                )
    else:
        # ── Multi-thread: pre-batch conversations, then parallel ──────
        batches = list(_batch_docs(doc_iter, args.max_docs))
        if args.max_samples:
            batches = batches[:max(0, args.max_samples - samples_done)]

        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {}
            for idx, batch in enumerate(batches):
                futures[pool.submit(
                    _worker_task, idx, batch, tokenizer,
                    args.api_base, api_key, args.api_protocol, args.model,
                    args.max_qa_tokens, args.max_tokens_per_call,
                    args.temperature, question_pool,
                    args.max_docs, args.max_group_size, args.max_group_query_num,
                )] = idx

            with open(out_path, mode) as out:
                results = {}
                remaining_futures = set(futures.keys())
                while remaining_futures:
                    done, remaining_futures = as_completed(
                        remaining_futures), set()
                    for future in done:
                        idx, data = future.result()
                        results[idx] = data

                    # Write in order as batches complete
                    while completed in results:
                        data = results.pop(completed)
                        completed += 1
                        if data is None:
                            errors += 1
                            continue
                        samples_done += 1
                        lines_read += len(batches[completed - 1])
                        out.write(json.dumps(data, ensure_ascii=False) + "\n")
                        out.flush()
                        state_path.write_text(
                            json.dumps({
                                "samples_done": samples_done,
                                "lines_read": lines_read,
                            })
                        )
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        n = data.get("num_turns", "?")
                        nd = data.get("num_docs", "?")
                        tk = data.get("assistant_token_len", "?")
                        print(
                            f"  [{completed}]  turns={n}  docs={nd}  "
                            f"tokens={tk}  {rate:.2f} conv/s"
                        )

    print(f"\nDone.  Successful: {completed}  Errors: {errors}")


if __name__ == "__main__":
    main()
