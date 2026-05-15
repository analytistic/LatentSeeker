"""LatentSeeker trainer builder."""

from typing import Any

from transformers import Trainer, TrainingArguments

from .callback import CurriculumCallback
from .collator import DynamicCompressCollator


class LatentSeekerTrainer(Trainer):
    """Logs svd_loss from LatentSeekerCausalLMOutput to TensorBoard."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        svd = getattr(outputs, "svd_loss", None)
        self._current_svd = svd.detach().cpu().item() if svd is not None else None
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self._current_svd is not None:
            logs["svd_loss"] = self._current_svd
        super().log(logs, start_time)


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

    trainer = LatentSeekerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        processing_class=processor,
    )
    return trainer
