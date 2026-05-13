"""Preprocess SQuAD → mapped Arrow dataset (single process, run once).

Usage:
    python -m src.dataset.preprocess_squad \
        --input data/squad \
        --output data/squad/processed_squad
"""

from datasets import load_dataset


def squad_to_messages(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "longtext", "longtext": example["context"]},
                    {"type": "text", "text": f"{example['question']}\nPlease answer the question concisely."},
                ],
            },
        ],
    }


def flatten_answers(example):
    example["answers"] = list(example["answers"]["text"])
    return example


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.input)

    if args.max_samples is not None:
        for split in dataset:
            dataset[split] = dataset[split].select(range(min(len(dataset[split]), args.max_samples)))

    # Apply map per-split to avoid DatasetDict.map() merging content dict keys (datasets 4.8.2)
    for split_name in list(dataset.keys()):
        ds = dataset[split_name]
        ds = ds.map(flatten_answers)
        ds = ds.map(squad_to_messages, remove_columns=["context"])
        dataset[split_name] = ds

    dataset.save_to_disk(args.output)
    print(f"Saved {len(dataset['train'])} train, {len(dataset['validation'])} validation samples to {args.output}")
