"""LatentSeeker trainer builder."""

from typing import Any

from transformers import Trainer, TrainingArguments

from .callback import CurriculumCallback
from .collator import DynamicCompressCollator


def build_trainer(
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any = None,
    args: TrainingArguments | None = None,
    compress_stages: list[tuple[float, int]] | None = None,
) -> Trainer:
    """Build a Trainer with LatentSeeker-specific collator and callbacks.

    Args:
        model: LatentSeekerForConditionalGeneration.
        processor: LatentSeekerProcessor.
        train_dataset: Dataset with "messages" column.
        compress_stages: Curriculum stages [(progress, compress_ratio), ...].
            If None, compress_ratio defaults to 8 throughout training.
    """
    collator = DynamicCompressCollator(
        processor=processor,
        vocab_size=model.config.text_config.vocab_size,
    )

    callbacks = []
    if compress_stages:
        callbacks.append(CurriculumCallback(compress_stages, collator=collator))

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
