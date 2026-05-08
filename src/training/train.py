"""LatentSeeker pre-training: repeat task with curriculum compression."""

import argparse
import sys

import yaml
from transformers import HfArgumentParser

from src.dataset.get_wiki import get_wiki
from src.models.LatentSeeker.modeling_LatentSeeker import (
    LatentSeekerForConditionalGeneration,
)
from src.models.LatentSeeker.configuration_LatentSeeker import LatentSeekerConfig
from src.models.LatentSeeker.processing_LatentSeeker import LatentSeekerProcessor
from src.utils.arguments import DataArgs, ModelArgs, LatentSeekerTrainingArguments
from src.utils.freeze import apply_freeze
from transformers import PreTrainedConfig

from .trainer import build_trainer



def _parse_args(
    config_path: str | None = None,
) -> tuple[LatentSeekerTrainingArguments, ModelArgs, DataArgs, dict | None]:
    """Parse args: YAML as defaults, CLI args override."""
    parser = HfArgumentParser((LatentSeekerTrainingArguments, ModelArgs, DataArgs))

    if config_path:
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}

        # Extract model config overrides (not a CLI arg)
        model_config_override = yaml_config.pop("model_config", None)

        flat = []
        for k, v in yaml_config.items():
            if v is not None:
                flat.append(f"--{k}")
                flat.append(str(v))

        # Strip --config_path from CLI overrides since HF parser doesn't know it
        cli = sys.argv[1:]
        for i, arg in enumerate(cli):
            if arg == "--config_path":
                cli = cli[:i] + cli[i + 2:]
                break

        train_args, model_args, data_args = parser.parse_args_into_dataclasses(
            args=flat + cli
        )
    else:
        model_config_override = None
        train_args, model_args, data_args = parser.parse_args_into_dataclasses()

    return train_args, model_args, data_args, model_config_override


def train(config_path: str | None = None):
    if config_path is None:
        p = argparse.ArgumentParser()
        p.add_argument("--config_path", default=None)
        parsed, _ = p.parse_known_args()
        config_path = parsed.config_path

    train_args, model_args, data_args, model_config_override = _parse_args(config_path)

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

    model = LatentSeekerForConditionalGeneration.init_from_pretrained(
        model_args.model_name,
        config=config,
    )

    apply_freeze(model, train_args.freeze_modules)

    dataset = get_wiki(data_args.data_path, max_samples=data_args.max_samples or 1000)

    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=dataset,
        args=train_args,
        compress_stages=train_args.compress_stages,
    )

    trainer.train()


if __name__ == "__main__":
    train()
