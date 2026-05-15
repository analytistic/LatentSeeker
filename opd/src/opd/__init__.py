"""OPD — On-Policy Distillation for LLMs.

Structure:
  RolloutWorker → Buffer → Trainer
  (student       (cache)  (teacher score + KL loss + backward)
   generate)

Usage:
    from opd import OPDTrainer, RolloutWorker, Buffer
"""

from .rollout import RolloutWorker, RolloutOutput
from .buffer import Buffer, BufferItem
from .trainer import OPDTrainer

__all__ = ["OPDTrainer", "RolloutWorker", "RolloutOutput", "Buffer", "BufferItem"]
