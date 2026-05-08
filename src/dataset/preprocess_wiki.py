"""Preprocess wiki JSONL → mapped Arrow dataset (single process, run once)."""
from datasets import load_dataset


def get_repeat_data(example):
    text = example["text"].strip()
    if not text or len(text.split()) < 20:
        return {"messages": None}
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "longtext", "longtext": text},
                    {"type": "text", "text": "Please repeat the longtext."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        ]
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/wiki/wiki.jsonl")
    parser.add_argument("--output", default="data/wiki/processed_wiki")
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.input, split="train")
    dataset = dataset.map(get_repeat_data, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x["messages"] is not None)
    dataset.save_to_disk(args.output)
    print(f"Saved {len(dataset)} samples to {args.output}")
