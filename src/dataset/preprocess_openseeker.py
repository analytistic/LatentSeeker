"""Preprocess openseeker search trajectory JSONL into Arrow dataset.

Reads the JSONL produced by openseeker pipeline and converts ``trajectory``
to ``messages`` column for training. Tool response messages (those starting
with ``<tool_response>``) are converted to ``longtext`` content blocks so
the LatentSeeker encoder compresses them.

Usage:
    python -m src.dataset.preprocess_openseeker \\
        --input data/openseeker/openseeker_v1_data.jsonl \\
        --output data/openseeker/processed_openseeker
"""

from __future__ import annotations

import argparse

from datasets import load_dataset, Features
from datasets.features import Sequence, Json, Value

TOOL_RESPONSE_PREFIX = "<tool_response>"


def convert_trajectory(trajectory: list) -> list:
    """Convert tool response messages to longtext blocks.

    - User messages starting with ``<tool_response>`` → longtext content block.
    - All other messages keep their plain text content.
    """
    messages = []
    for msg in trajectory:
        role = msg["role"]
        content = msg["content"]

        if (
            role == "user"
            and isinstance(content, str)
            and content.startswith(TOOL_RESPONSE_PREFIX)
        ):
            messages.append({
                "role": "user",
                "content": [{"type": "longtext", "longtext": content}],
            })
        else:
            messages.append({"role": role, "content": content})

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess openseeker trajectory data into Arrow dataset"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.input, split="train")
    if args.max_samples:
        ds = ds.select(range(args.max_samples))

    ds = ds.map(
        lambda ex: {
            "messages": convert_trajectory(ex["trajectory"]),
            "num_tool_calls": ex["number of tool calls"],
        },
        remove_columns=ds.column_names,
        features=Features({
            "messages": Sequence(Json(decode=True)),
            "num_tool_calls": Value("int64"),
        }),
    )

    ds.save_to_disk(args.output)
    print(f"Saved {len(ds)} samples to {args.output}")


if __name__ == "__main__":
    main()
