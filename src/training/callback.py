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

    def __init__(self, stages: list[tuple[float, int]], collator=None):
        """
        stages: sorted list of (progress_threshold, compress_ratio).
                progress is global_step / max_steps ∈ [0, 1].
        """
        self.stages = sorted(stages, key=lambda x: x[0])
        self._current_ratio = None
        self._collator = collator
        if collator is not None:
            collator.compress_ratio = self.stages[0][1]

    def on_step_begin(self, args, state, control, **kwargs):
        max_steps = state.max_steps
        if max_steps <= 0:
            return
        progress = state.global_step / max_steps
        collator = self._collator
        if collator is None:
            return

        # Find the highest threshold ≤ current progress
        ratio = self.stages[-1][1]
        for threshold, r in self.stages:
            if progress >= threshold:
                ratio = r

        if ratio != self._current_ratio:
            self._current_ratio = ratio
            if state.is_world_process_zero:
                print(
                    f"[CurriculumCallback] Step {state.global_step} ({progress*100:.1f}%): "
                    f"switching compress_ratio → {ratio}"
                )

        collator.compress_ratio = ratio
