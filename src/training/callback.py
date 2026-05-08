"""Curriculum learning callback for LatentSeeker.

Adjusts compress_ratio in the data collator based on training progress.
"""

from transformers import TrainerCallback


class CurriculumCallback(TrainerCallback):
    """
    Progressively increases compress_ratio during training.

    Example:
        stages = [(0, 2), (0.1, 8), (0.5, 32)]
        → 0-10%:  ratio=2  (nearly copy)
        → 10-50%: ratio=8  (moderate compression)
        → 50-100%: ratio=32 (extreme compression)
    """

    def __init__(self, stages: list[tuple[float, int]]):
        """
        stages: sorted list of (progress_threshold, compress_ratio).
                progress is global_step / max_steps ∈ [0, 1].
        """
        self.stages = sorted(stages, key=lambda x: x[0])

    def on_step_begin(self, args, state, control, **kwargs):
        progress = state.global_step / max(args.max_steps, 1)
        collator = kwargs.get("data_collator")
        if collator is None:
            return

        # Find the highest threshold ≤ current progress
        ratio = self.stages[-1][1]
        for threshold, r in self.stages:
            if progress >= threshold:
                ratio = r

        collator.compress_ratio = ratio
