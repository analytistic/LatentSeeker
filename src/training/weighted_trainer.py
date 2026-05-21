"""Trainer with weighted multi-dataset sampling for multi-task training."""
from __future__ import annotations

import random

from torch.utils.data import Sampler
from transformers import Trainer


class BalancedMultiTaskSampler(Sampler[int]):
    """One-balanced-shuffle-per-epoch multi-task sampler.

    For each epoch:
    1. Take the dataset with the fewest samples as the *anchor*.
    2. Total epoch size = anchor_samples / anchor_weight.
    3. For each dataset: randomly draw ``ceil(total_size * weight)`` samples
       (without replacement when possible, with replacement otherwise).
    4. Concatenate all drawn indices into one list, shuffle once, iterate.
    """

    def __init__(
        self,
        dataset_lengths: dict[str, int],
        dataset_weights: dict[str, float],
    ):
        self.dataset_lengths = dataset_lengths
        self.dataset_weights = dataset_weights

        # Pre-compute per-dataset offsets in the concatenated dataset
        names = list(dataset_lengths.keys())
        self.offsets: dict[str, int] = {}
        cum = 0
        for name in names:
            self.offsets[name] = cum
            cum += dataset_lengths[name]

        # Total epoch size: anchor = dataset with smallest (length / weight)
        anchor_name = min(
            dataset_lengths,
            key=lambda k: dataset_lengths[k] / max(dataset_weights.get(k, 1.0), 1e-8),
        )
        anchor_len = dataset_lengths[anchor_name]
        anchor_weight = max(dataset_weights.get(anchor_name, 1.0), 1e-8)
        self.total_size = int(anchor_len / anchor_weight)

    def __iter__(self):
        all_indices = []
        for name, length in self.dataset_lengths.items():
            n = round(self.total_size * self.dataset_weights.get(name, 1.0))
            offset = self.offsets[name]
            if n <= length:
                chosen = random.sample(range(length), n)
            else:
                chosen = random.sample(range(length), length)
                chosen += random.choices(range(length), k=n - length)
            all_indices.extend(offset + i for i in chosen)

        random.shuffle(all_indices)
        return iter(all_indices)

    def __len__(self):
        return self.total_size


class WeightedMultiTaskTrainer(Trainer):
    """Trainer that samples from concatenated datasets with per-dataset weights.

    Usage::

        dataset, lengths = get_weighted_mixer({
            "wiki": wiki_ds,
            "qa": qa_ds,
        })
        trainer = WeightedMultiTaskTrainer(
            ...,
            train_dataset=dataset,
            dataset_lengths=lengths,
            dataset_weights={"wiki": 0.5, "qa": 0.5},
        )
    """

    def __init__(self, *, dataset_lengths: dict[str, int], dataset_weights: dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.dataset_lengths = dataset_lengths
        self.dataset_weights = dataset_weights

    def _get_train_sampler(self, train_dataset=None):
        return BalancedMultiTaskSampler(self.dataset_lengths, self.dataset_weights)
