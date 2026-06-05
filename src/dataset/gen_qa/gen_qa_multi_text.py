#!/usr/bin/env python3
"""Generate multi-question QA data from multiple longtext documents.

Each sample contains multiple documents in a single round, with multiple
questions generated about the document collection.

Input format (JSONL, one sample per line):
    {"documents": ["doc1 text...", "doc2 text...", ...]}

Output format:
    messages: [
        {role: user, content: [longtext(doc1), longtext(doc2), ..., text(q1)]},
        {role: assistant, reasoning_content(r1), content(text(a1))},
        {role: user, content: [text(q2)]},
        {role: assistant, reasoning_content(r2), content(text(a2))},
        ...
    ]

Usage:
    python -m src.dataset.gen_qa.gen_qa_multi_text \\
        --input data/multi_docs.jsonl \\
        --output data/qa_multi.jsonl \\
        --num-questions 5 \\
        --max-qa-tokens 3000

    python -m src.dataset.gen_qa.gen_qa_multi_text \\
        --input data/multi_docs.jsonl \\
        --output data/qa_multi.jsonl \\
        --num-questions 8 \\
        --max-workers 16
"""

from __future__ import annotations

import argparse
import json
import os
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
    format_history,
    load_progress,
    parse_one_turn,
    parse_type_spec,
    sample_type,
    save_progress,
)

# ── Prompts ─────────────────────────────────────────────────────────────

ADDITIVE_PROMPT = """Documents:
{document}
{history_section}
Now generate the next QA turn about the above document(s).

Question type: {type_instruction}

Output exactly in this format:
Question: <question>
Reasoning: <step-by-step reasoning>
Answer: <concise answer>"""


# ── Message building ────────────────────────────────────────────────────

def build_messages(
    doc_texts: list[str], turns: list[dict[str, str]]
) -> list[dict[str, Any]] | None:
    """Build LatentSeeker messages with multiple longtext blocks.

    First user message: [longtext(doc1), longtext(doc2), ..., text(q1)]
    Subsequent user messages: [text(qN)]
    """
    if not turns or not doc_texts:
        return None

    # First user message: all longtexts + first question
    first_content: list[dict[str, str]] = [
        {"type": "longtext", "longtext": doc} for doc in doc_texts
    ]
    first_content.append({"type": "text", "text": turns[0]["question"]})

    msgs = [{"role": "user", "content": first_content}]

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


# ── Document-level generation ───────────────────────────────────────────

def process_doc(
    doc_texts: list[str],
    tokenizer: AutoTokenizer,
    api_base: str,
    api_key: str,
    protocol: Literal["anthropic", "openai"],
    model: str,
    budget_tokens: int,
    max_tokens_per_call: int,
    temperature: float,
    question_pool: list[str],
    num_questions: int,
) -> tuple[list[dict[str, str]], int, bool, float] | None:
    """Generate turns additively, counting real tokens.

    Returns ``(turns, total_assistant_tokens, budget_exceeded, api_time_s)``
    or ``None``.
    """
    if not doc_texts:
        return None

    # Format all documents for the prompt
    doc_section = "\n\n---\n\n".join(
        f"Document {i+1}:\n{doc}" for i, doc in enumerate(doc_texts)
    )

    turns: list[dict[str, str]] = []
    total_tokens = 0
    api_time_s = 0.0
    max_retries = 3
    budget_exceeded = False
    last_type: str | None = None

    while len(turns) < num_questions:
        # 1. Sample question type
        qtype_name = sample_type(question_pool, last_type)
        last_type = qtype_name
        qtype = QUESTION_TYPE_DEFS[qtype_name]
        type_instruction = qtype["instruction"]

        # 2. Build prompt
        history = format_history(turns)
        prompt = ADDITIVE_PROMPT.format(
            document=doc_section,
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
        msgs = build_messages(doc_texts, candidate_turns)
        if msgs is None:
            continue
        candidate_total = len(
            tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                multi_turn_reasoning=True,
            )["input_ids"]
        )

        # 6. If budget exceeded: keep this turn but stop
        if candidate_total > budget_tokens:
            budget_exceeded = True
            turns.append(turn)
            break

        turns.append(turn)
        total_tokens = candidate_total

    return (turns, total_tokens, budget_exceeded, api_time_s) if turns else None


# ── Worker ──────────────────────────────────────────────────────────────

def _worker(args: tuple) -> tuple[int, tuple | None]:
    (
        idx,
        doc_texts,
        tokenizer,
        api_base,
        api_key,
        protocol,
        model,
        budget_tokens,
        max_tokens_per_call,
        temperature,
        question_pool,
        num_questions,
    ) = args
    result = process_doc(
        doc_texts, tokenizer, api_base, api_key, protocol, model,
        budget_tokens, max_tokens_per_call, temperature,
        question_pool, num_questions,
    )
    if result:
        return idx, result
    return idx, None


# ── CLI ─────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate multi-question QA data from multiple longtext documents"
    )
    add_common_args(p)
    p.add_argument(
        "--num-questions",
        type=int,
        default=3,
        help="Number of questions to generate per sample (may be fewer if budget exceeded)",
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
    print(f"Num questions per sample: {args.num_questions}")
    from collections import Counter
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
                doc_texts = dataset[idx]["documents"]
                result = process_doc(
                    doc_texts, tokenizer,
                    args.api_base, api_key, args.api_protocol, args.model,
                    args.max_qa_tokens, args.max_tokens_per_call,
                    args.temperature, question_pool, args.num_questions,
                )
                turns_info = None
                if result:
                    turns, total_tokens, budget_exceeded, api_time_s = result
                    msgs = build_messages(doc_texts, turns)
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
                save_progress(state_path, done)
                n = turns_info["num_turns"] if turns_info else 0
                eta = _eta(time.time() - start_time, completed, len(remaining))
                print(
                    f"  [{completed}/{len(remaining)}]  sample {idx}  "
                    f"ok={completed}  err={errors}  "
                    f"turns={n}  api={api_time_s:.1f}s  eta={eta}"
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
                    args.num_questions,
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
                        turns, total_tokens, budget_exceeded, api_time_s = result
                        doc_texts = dataset[idx]["documents"]
                        msgs = build_messages(doc_texts, turns)
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
                    save_progress(state_path, done)
                    n = turns_info["num_turns"] if turns_info else 0
                    eta = _eta(time.time() - start_time, completed, len(remaining))
                    print(
                        f"  [{completed}/{len(remaining)}]  sample {idx}  "
                        f"ok={completed}  err={errors}  "
                        f"turns={n}  api={api_time_s:.1f}s  eta={eta}"
                    )

    print(f"\nDone.  Successful: {completed}  Errors: {errors}")


if __name__ == "__main__":
    main()
