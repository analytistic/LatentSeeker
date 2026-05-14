"""Dynamic compression collator for LatentSeeker training.

Tokenizes messages on-the-fly with configurable compress_ratio,
enabling curriculum learning without pre-computing multiple tokenized copies.
"""

import multiprocessing as mp
from typing import Any

import torch


class DynamicCompressCollator:
    """Collator that tokenizes with dynamic compress_ratio.

    Uses multiprocessing.Value for compress_ratio so that changes made
    by CurriculumCallback in the main process are visible to DataLoader
    worker processes (num_workers > 0).

    Usage:
        collator = DynamicCompressCollator(processor)
        collator.compress_ratio = 32  # adjusted by callback during training
    """

    def __init__(self, processor, vocab_size=None, compress_ratio=8):
        self.processor = processor
        self.vocab_size = vocab_size
        self._ratio = mp.Value('d', compress_ratio)

    @property
    def compress_ratio(self) -> int | float:
        return self._ratio.value

    @compress_ratio.setter
    def compress_ratio(self, value: int | float):
        self._ratio.value = value

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        out = self.processor.apply_chat_template(
            [item["messages"] for item in batch],
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            compress_ratio=self.compress_ratio,
            return_tensors="pt",
            padding=True,
        )

        # Remap OOB tokens for small vocab debug configs
        if self.vocab_size is not None and self.vocab_size < self.processor.longtext_token_id:
            pad_slot = self.vocab_size - 1
            nonpad_range = pad_slot
            for key in ("input_ids", "longtext_input_ids"):
                t = out[key]
                is_pad = t == self.processor.longtext_token_id
                t[~is_pad] = t[~is_pad] % nonpad_range
                t[is_pad] = pad_slot

        # Labels: -100 for non-assistant positions
        labels = out["input_ids"].clone()
        labels[~out["assistant_masks"].bool()] = -100

        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
            "labels": labels,
            "longtext_input_ids": out["longtext_input_ids"],
            "longtext_cu_seqlens": out["longtext_cu_seqlens"],
            "longtext_num_tokens": out["longtext_num_tokens"],
        }
