"""Load preprocessed multi-turn QA Arrow dataset."""

from datasets import load_from_disk


def get_qa(data_path: str, max_samples: int | None = None):
    ds = load_from_disk(data_path)
    if max_samples is not None:
        ds = ds.select(range(max_samples))
    return ds
