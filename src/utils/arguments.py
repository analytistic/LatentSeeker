from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class DataArgs:
    data_path: str | None = field(default=None, metadata={"help": "Single dataset path (backward compat). Ignored if datasets is set."})
    max_samples: int | None = field(default=None, metadata={"help": "Limit dataset size for debugging (backward compat)."})
    datasets: dict | None = field(
        default=None,
        metadata={"help": "Multi-dataset config: {name: {path, weight, max_samples}}. "
                         "e.g. '{\"wiki\": {\"path\": \"data/wiki/processed_wiki\", \"weight\": 0.5}}'"},
    )


@dataclass
class ModelArgs:
    model_name: str = field(
        default="src/models/LatentSeeker",
        metadata={"help": "Path to the base model directory."},
    )
    model_ckpt_path: str | None = field(
        default=None,
        metadata={"help": "Trained checkpoint path. If set, loads weights directly via from_pretrained "
                         "instead of init_from_pretrained (skips embed/layer initialization)."},
    )
    model_cache_dir: str = field(
        default="",
        metadata={"help": "Cache directory for downloaded models."},
    )


@dataclass
class LatentSeekerTrainingArguments(TrainingArguments):
    """LatentSeeker-specific training arguments with sensible defaults."""

    compress_stages: list[tuple[float, int]] = field(
        default_factory=lambda: [(0, 2), (0.1, 8), (0.5, 32)],
        metadata={"help": "Curriculum stages: [(threshold, compress_ratio), ...]"},
    )

    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Must be False — collator needs 'messages' column."},
    )

    trainer: str = field(
        default="Trainer",
        metadata={"help": "Trainer class name. Registered: Trainer, WeightedMultiTaskTrainer, OPSDTrainer"},
    )

    freeze_modules: list[str] | None = field(
        default=None,
        metadata={"help": "Module paths to freeze. e.g. ['model.language_model']"},
    )

    bf16: bool = field(default=True)

