"""Preprocess multi-turn QA JSONL into Arrow dataset.

Reads the JSONL produced by ``gen_qa.py`` and converts it to Arrow format.
Uses the ``num_turns`` field from JSONL to control truncation — keeps the
first N fitting turns (each Q+R+A pair).

Usage:
    python -m src.dataset.preprocess_qa \\
        --input data/wiki/qa.jsonl \\
        --output data/wiki/processed_qa \\
        --max-turns 5
"""

from __future__ import annotations

import argparse

from datasets import load_dataset


def truncate_by_turns(messages: list, num_turns: int, max_turns: int) -> list:
    """Keep only the first ``max_turns`` fitting turns.

    Args:
        messages: Full message list (may have ``num_turns`` or ``num_turns + 1``
            assistant turns if budget was exceeded in gen_qa).
        num_turns: Fitting turn count from the JSONL.
        max_turns: Desired maximum fitting turns.

    Returns:
        Truncated messages with at most ``max_turns`` assistant turns.
    """
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
        help="Truncate to first N fitting turns per sample. "
             "Default: keep all turns.",
    )
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.input, split="train")
    if args.max_samples:
        ds = ds.select(range(args.max_samples))

    # Keep only messages & num_turns, re-infer schema (like wiki's remove_columns)
    keep_cols = ["messages", "num_turns"]

    def _process(ex, idx):
        if args.max_turns is not None:
            nt = ex.get("num_turns")
            if nt is not None:
                return {
                    "messages": truncate_by_turns(
                        ex["messages"], nt, args.max_turns
                    ),
                    "num_turns": min(nt, args.max_turns),
                }
        return {k: ex[k] for k in keep_cols}

    ds = ds.map(_process, remove_columns=ds.column_names, with_indices=True)

    ds.save_to_disk(args.output)
    print(f"Saved {len(ds)} samples to {args.output}")


if __name__ == "__main__":
    main()
