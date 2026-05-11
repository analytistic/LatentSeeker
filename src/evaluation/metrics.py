"""Evaluation metrics for LatentSeeker paraphrase/reconstruction tasks."""

import math
from collections.abc import Callable
from collections import Counter


class Metrics:
    """Collection of evaluation metrics for text reconstruction quality.

    Parameters
    ----------
    tokenizer : callable, optional
        A tokenizer with an ``encode`` method (e.g. ``PreTrainedTokenizer.encode``).
        When set, BLEU's n-grams are computed over the tokenizer's BPE ids
        (converted to strings) instead of whitespace-split words.  This avoids
        a second tokenisation pass with a different scheme and keeps the
        metric consistent with the model's own vocabulary.
    """

    def __init__(self, tokenizer: Callable | None = None):
        self.tokenizer = tokenizer

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer is not None:
            return [str(id_) for id_ in self.tokenizer.encode(text)]
        return text.lower().split()

    @staticmethod
    def _ngrams(tokens: list[str], n: int) -> list[tuple]:
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    # ------------------------------------------------------------------
    # Token-level accuracy
    # ------------------------------------------------------------------

    def token_accuracy(self, generated_ids: list[int], reference_ids: list[int]) -> dict:
        """Position-by-position BPE-token accuracy.

        Parameters
        ----------
        generated_ids:
            Token ids produced by the model (after trimming the prompt).
        reference_ids:
            Token ids of the reference text (encoded with the same tokenizer).

        Returns
        -------
        dict with keys ``accuracy`` (float 0-1), ``matches``, ``total``.
        """
        min_len = min(len(generated_ids), len(reference_ids))
        if min_len == 0:
            return {"accuracy": 0.0, "matches": 0, "total": 0}

        matches = sum(1 for i in range(min_len) if generated_ids[i] == reference_ids[i])
        return {
            "accuracy": matches / min_len,
            "matches": matches,
            "total": min_len,
        }

    # ------------------------------------------------------------------
    # BLEU
    # ------------------------------------------------------------------

    def bleu(self, hypothesis: str, reference: str, max_n: int = 4) -> float:
        """Sentence-level BLEU (1-4 gram).

        Uses the configured tokenizer for tokenisation (BPE ids by default,
        whitespace fallback).  Returns a score in [0, 100].
        """
        hyp = self._tokenize(hypothesis)
        ref = self._tokenize(reference)

        hyp_len = len(hyp)
        ref_len = len(ref)
        if hyp_len == 0 or ref_len == 0:
            return 0.0

        # Brevity penalty
        bp = min(1.0, math.exp(1 - ref_len / hyp_len)) if hyp_len < ref_len else 1.0

        # Modified n-gram precision
        precisions = []
        for n in range(1, max_n + 1):
            if hyp_len < n:
                precisions.append(0.0)
                continue
            hyp_ngrams = Counter(self._ngrams(hyp, n))
            ref_ngrams = Counter(self._ngrams(ref, n))
            matches = sum(
                min(count, ref_ngrams.get(ng, 0))
                for ng, count in hyp_ngrams.items()
            )
            total = hyp_len - n + 1
            precisions.append(matches / total if total > 0 else 0.0)

        if any(p == 0 for p in precisions):
            return 0.0

        log_avg = sum(math.log(p) for p in precisions) / max_n
        return bp * math.exp(log_avg) * 100

    # ------------------------------------------------------------------
    # Batched helpers
    # ------------------------------------------------------------------

    def batch_token_accuracy(
        self,
        generated_batch: list[list[int]],
        reference_batch: list[list[int]],
    ) -> dict:
        """Average token accuracy across a batch."""
        total_matches = 0
        total_tokens = 0
        per_sample = []
        for gen, ref in zip(generated_batch, reference_batch):
            result = self.token_accuracy(gen, ref)
            per_sample.append(result)
            total_matches += result["matches"]
            total_tokens += result["total"]
        return {
            "accuracy": total_matches / total_tokens if total_tokens else 0.0,
            "per_sample": per_sample,
        }

    def batch_bleu(
        self,
        hypotheses: list[str],
        references: list[str],
        max_n: int = 4,
    ) -> dict:
        """Average BLEU across a batch."""
        scores = [self.bleu(h, r, max_n) for h, r in zip(hypotheses, references)]
        return {
            "bleu": sum(scores) / len(scores) if scores else 0.0,
            "per_sample": scores,
        }
