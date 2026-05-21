"""Weighted multi-dataset mixing for multi-task training."""
from __future__ import annotations

from datasets import concatenate_datasets, Dataset
from torch.utils.data import WeightedRandomSampler


def get_weighted_mixer(
    datasets: dict[str, Dataset],
) -> tuple[Dataset, dict[str, int]]:
    """Concatenate multiple datasets and return per-dataset lengths.

    Args:
        datasets: Mapping of ``{name: dataset}``.

    Returns:
        ``(concatenated_dataset, {name: length})``
    """
    lengths = {k: len(v) for k, v in datasets.items()}
    combined = concatenate_datasets(list(datasets.values()))
    return combined, lengths
