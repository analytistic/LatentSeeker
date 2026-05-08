from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class DataArgs:
    data_name: str = field(default="wiki", metadata={"help": "The name of the dataset to use (via the datasets library)."})
    data_path: str = field(default="data/wiki/wiki.jsonl", metadata={"help": "Path or name of the dataset; if None, defaults to data_name."})
    split: str = field(default="train", metadata={"help": "The split of the dataset to use."})
    max_samples: int | None = field(default=None, metadata={"help": "Limit dataset size for debugging."})


@dataclass
class ModelArgs:
    model_name: str = field(
        default="src/models/LatentSeeker",
        metadata={"help": "Path to the model directory."},
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

    bf16: bool = field(default=True)

