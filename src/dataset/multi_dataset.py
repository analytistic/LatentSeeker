"""Weighted multi-dataset mixing for multi-task training."""
from __future__ import annotations

from datasets import concatenate_datasets, Dataset


def get_weighted_mixer(
    datasets: dict[str, Dataset],
) -> tuple[Dataset, dict[str, int], list[str]]:
    """Concatenate multiple datasets and return per-dataset lengths + order.

    Returns an ordered list of dataset names so callers (e.g. the multi-task
    sampler) can compute offsets in the same order as the concatenation.

    Args:
        datasets: Mapping of ``{name: dataset}``.

    Returns:
        ``(concatenated_dataset, {name: length}, [name1, name2, ...])``
    """
    names = list(datasets.keys())
    lengths = {k: len(v) for k, v in datasets.items()}
    combined = concatenate_datasets([datasets[n] for n in names])
    return combined, lengths, names
