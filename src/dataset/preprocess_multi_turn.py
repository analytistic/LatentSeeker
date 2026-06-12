"""Preprocess multi-turn QA JSONL into Arrow dataset.

Reads the JSONL produced by ``gen_qa_multi_turn.py`` and converts it to
Arrow format for training.

Usage:
    python -m src.dataset.preprocess_multi_turn \\
        --input data/qa_multi_turn.jsonl \\
        --output data/multi_turn/processed_gen_multi_turn
"""

from __future__ import annotations

import argparse

from datasets import load_dataset, Features
from datasets.features import Sequence, Json, Value


def truncate_by_turns(messages: list, num_turns: int, max_turns: int) -> list:
    """Keep only the first ``max_turns`` fitting turns."""
    keep = min(max_turns, num_turns)
    assistant_idxs = [
        i for i, m in enumerate(messages) if m["role"] == "assistant"
    ]
    if len(assistant_idxs) <= keep:
        return messages
    last = assistant_idxs[keep - 1] + 1
    return messages[:last]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess multi-turn QA JSONL into Arrow dataset"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Truncate to first N fitting turns per sample. Default: keep all.",
    )
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.input, split="train")
    if args.max_samples:
        ds = ds.select(range(args.max_samples))

    if args.max_turns is not None:
        def _truncate(ex, idx):
            nt = ex.get("num_turns")
            if nt is None:
                return ex
            ex["messages"] = truncate_by_turns(
                ex["messages"], nt, args.max_turns
            )
            ex["num_turns"] = min(nt, args.max_turns)
            return ex

        ds = ds.map(_truncate, with_indices=True)

    # Keep only needed columns with consistent schema
    ds = ds.map(
        lambda ex: {
            "messages": ex["messages"],
            "num_turns": ex["num_turns"],
        },
        remove_columns=ds.column_names,
        features=Features({
            "messages": Sequence(Json(decode=True)),
            "num_turns": Value("int64"),
        }),
    )

    ds.save_to_disk(args.output)
    print(f"Saved {len(ds)} samples to {args.output}")


if __name__ == "__main__":
    main()
