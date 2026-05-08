from datasets import load_from_disk


def get_wiki(path, max_samples=None):
    dataset = load_from_disk(path)

    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    return dataset
