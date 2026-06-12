"""LatentSeeker training entry point — dispatches to trainer class by name."""

import argparse
import json
import sys
from typing import Any

import yaml
from datasets import load_from_disk
from transformers import HfArgumentParser

from src.dataset.get_wiki import get_wiki
from src.dataset.multi_dataset import get_weighted_mixer
from src.models.LatentSeeker.modeling_LatentSeeker import (
    LatentSeekerForConditionalGeneration,
)
from src.models.LatentSeeker.configuration_LatentSeeker import LatentSeekerConfig
from src.models.LatentSeeker.processing_LatentSeeker import LatentSeekerProcessor
from src.utils.arguments import DataArgs, ModelArgs, LatentSeekerTrainingArguments
from src.utils.freeze import apply_freeze
from transformers import PreTrainedConfig

from .callback import CurriculumCallback
from .collator import DynamicCompressCollator
from .opd_trainer import OPSDTrainer
from .weighted_trainer import WeightedMultiTaskTrainer
from .trainer import Trainer


# ── Trainer registry ────────────────────────────────────────────────────────

TRAINER_REGISTRY: dict = {
    "Trainer": Trainer,
    "WeightedMultiTaskTrainer": WeightedMultiTaskTrainer,
    "OPSDTrainer": OPSDTrainer,
}


def build_trainer(
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any = None,
    args: LatentSeekerTrainingArguments | None = None,
    compress_stages: list[tuple[float, int]] | None = None,
    dataset_lengths: dict[str, int] | None = None,
    dataset_weights: dict[str, float] | None = None,
    dataset_names: list[str] | None = None,
) -> Trainer:
    """Build a Trainer with LatentSeeker-specific collator and callbacks.

    Args:
        model: LatentSeekerForConditionalGeneration.
        processor: LatentSeekerProcessor.
        train_dataset: Dataset with "messages" column.
        compress_stages: Curriculum stages [(progress, compress_ratio), ...].
            If None, compress_ratio defaults to 8 throughout training.
        dataset_lengths: Per-dataset lengths for weighted sampling.
        dataset_weights: Per-dataset sampling weights.
        dataset_names: Ordered dataset names matching the combined dataset.
    """
    collator = DynamicCompressCollator(
        processor=processor,
        vocab_size=model.config.text_config.vocab_size,
    )

    callbacks = []
    if compress_stages:
        callbacks.append(CurriculumCallback(compress_stages, collator=collator))

    trainer_name = getattr(args, "trainer", None) or "Trainer"
    trainer_cls = TRAINER_REGISTRY.get(trainer_name)
    if trainer_cls is None:
        raise ValueError(
            f"Unknown trainer '{trainer_name}'. Available: {list(TRAINER_REGISTRY)}"
        )

    kwargs: dict[str, Any] = {}
    if dataset_lengths is not None:
        kwargs["dataset_lengths"] = dataset_lengths
        kwargs["dataset_weights"] = dataset_weights or {}
        kwargs["dataset_names"] = dataset_names or []

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        processing_class=processor,
        **kwargs,
    )
    return trainer


# ── Argument parsing ────────────────────────────────────────────────────────

def _parse_args(
    config_path: str | None = None,
) -> tuple[LatentSeekerTrainingArguments, ModelArgs, DataArgs, dict | None]:
    """Parse args: YAML as defaults, CLI args override."""
    parser = HfArgumentParser((LatentSeekerTrainingArguments, ModelArgs, DataArgs))

    if config_path:
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}

        # Extract params HF's parser can't handle
        model_config_override = yaml_config.pop("model_config", None)
        resume_from_checkpoint = yaml_config.pop("resume_from_checkpoint", None)
        datasets_config = yaml_config.pop("datasets", None)

        # HfArgumentParser can't handle nested list types like list[tuple[float, int]].
        complex_list_key = "compress_stages"
        complex_list_val = yaml_config.pop(complex_list_key, None)

        flat = []
        for k, v in yaml_config.items():
            if v is not None:
                flat.append(f"--{k}")
                if isinstance(v, list):
                    for item in v:
                        flat.append(json.dumps(item) if isinstance(item, (list, tuple)) else str(item))
                elif isinstance(v, dict):
                    flat.append(json.dumps(v))
                else:
                    flat.append(str(v))

        # Strip args HF parser doesn't know about
        cli = sys.argv[1:]
        for skip_key in ("--config_path", "--resume_from_checkpoint"):
            for i, arg in enumerate(cli):
                if arg == skip_key:
                    cli = cli[:i] + cli[i + 2:]
                    break
        # Also check if resume_from_checkpoint was passed via CLI
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--resume_from_checkpoint" and i + 2 <= len(sys.argv[1:]):
                resume_from_checkpoint = sys.argv[1:][i + 1]

        train_args, model_args, data_args = parser.parse_args_into_dataclasses(
            args=flat + cli
        )

        # Restore nested complex types that HfArgumentParser can't handle from CLI args
        if complex_list_val is not None:
            train_args.compress_stages = [tuple(item) for item in complex_list_val]
        if datasets_config is not None:
            data_args.datasets = datasets_config
    else:
        model_config_override = None
        resume_from_checkpoint = None
        train_args, model_args, data_args = parser.parse_args_into_dataclasses()

    return train_args, model_args, data_args, model_config_override, resume_from_checkpoint


def _load_datasets(data_args: DataArgs):
    """Load datasets from config.

    Returns ``(train_dataset, dataset_lengths, dataset_weights)`` if multi-dataset,
    otherwise ``(train_dataset, None, None)``.
    """
    if data_args.datasets:
        datasets = {}
        weights = {}
        for name, cfg in data_args.datasets.items():
            path = cfg["path"]
            max_s = cfg.get("max_samples", None)
            weight = cfg.get("weight", 1.0)
            ds = load_from_disk(path)
            if max_s is not None:
                ds = ds.select(range(min(len(ds), max_s)))
            datasets[name] = ds
            weights[name] = weight

        combined, lengths, names = get_weighted_mixer(datasets)
        return combined, lengths, weights, names

    # Fallback: single dataset
    dataset = get_wiki(data_args.data_path, max_samples=data_args.max_samples)
    return dataset, None, None, None


def train(config_path: str | None = None):
    if config_path is None:
        p = argparse.ArgumentParser()
        p.add_argument("--config_path", default=None)
        parsed, _ = p.parse_known_args()
        config_path = parsed.config_path

    train_args, model_args, data_args, model_config_override, resume_from_checkpoint = _parse_args(config_path)

    processor = LatentSeekerProcessor.from_pretrained(model_args.model_name)
    config = LatentSeekerConfig.from_pretrained(model_args.model_name)

    # Apply model config overrides for debug
    if model_config_override:
        for key, value in model_config_override.items():
            sub = getattr(config, key, None)
            if isinstance(sub, PreTrainedConfig):
                sub.update(value)
            else:
                setattr(config, key, value)
        # Small vocab debug: longtext_token_id = last vocab slot
        if config.text_config.vocab_size <= processor.longtext_token_id:
            config.longtext_token_id = config.text_config.vocab_size - 1

    if model_args.model_ckpt_path:
        model = LatentSeekerForConditionalGeneration.from_pretrained(
            model_args.model_ckpt_path,
            config=config,
        )
    else:
        model = LatentSeekerForConditionalGeneration.init_from_pretrained(
            model_args.model_name,
            config=config,
        )

    apply_freeze(model, train_args.freeze_modules)

    dataset, dataset_lengths, dataset_weights, dataset_names = _load_datasets(data_args)

    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=dataset,
        args=train_args,
        compress_stages=train_args.compress_stages,
        dataset_lengths=dataset_lengths,
        dataset_weights=dataset_weights,
        dataset_names=dataset_names,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    train()
