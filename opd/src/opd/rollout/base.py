"""Abstract base class for rollout backends.

A backend implements generate() to produce tokens + logits from prompts.
Backends handle their own initialization (device map, distributed config, etc.)
and expose only a uniform generate() interface to the trainer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RolloutOutput:
    """Single sample from a rollout step.

    `tokens` = prompt + response concatenated, split by `response_length`.
    `rollout_logits` shape = [response_length, vocab_size].
    """
    tokens: list[int] | torch.Tensor          # [prompt || response]
    response_length: int                      # where response starts in tokens
    rollout_logits: torch.Tensor              # [response_length, vocab_size]
    rollout_hidden_states: tuple | None = None
    loss_mask: list[int] | None = None        # 1 for loss positions, default: response only
    full_text: str = ""                       # Full prompt text for teacher scoring


class RolloutBackend(ABC):
    """Interface for rollout generation engines.

    Each backend (HF, vLLM, SGLang) implements this interface.
    Initialization is backend-specific — the base class only
    defines the generate() contract.

    Backends that need weight sync from training (multi-process setups)
    should override update(). Backends that hold GPU resources
    should override shutdown().
    """

    @abstractmethod
    def generate(
        self, batch: list[dict[str, Any]], **kwargs
    ) -> list[RolloutOutput]:
        """Generate responses for a batch of prompts.

        Args:
            batch: List of samples, each with at least "messages".
            **kwargs: Backend-specific overrides.

        Returns:
            List of RolloutOutput, one per sample * num_return_sequences.
        """
        ...

    def update(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        """Sync weights from training into the inference engine.

        Only needed for multi-process setups.
        Single-process backends (HF) can leave this as a no-op.
        """
        pass

    def shutdown(self) -> None:
        """Release resources (GPU memory, server connections, etc.)."""
        pass
