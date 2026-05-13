from datasets import load_from_disk


def load(path, split="validation", max_samples=None):
    dataset = load_from_disk(path)
    ds = dataset[split]

    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    return ds
