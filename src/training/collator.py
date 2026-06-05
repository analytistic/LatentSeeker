"""Dynamic compression collator for LatentSeeker training.

Tokenizes messages on-the-fly with configurable compress_ratio,
enabling curriculum learning without pre-computing multiple tokenized copies.
"""

import multiprocessing as mp
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


class DynamicCompressCollator:
    """Collator that tokenizes with dynamic compress_ratio.

    Uses multiprocessing.Value for compress_ratio so that changes made
    by CurriculumCallback in the main process are visible to DataLoader
    worker processes (num_workers > 0).

    Usage:
        collator = DynamicCompressCollator(processor)
        collator.compress_ratio = 32  # adjusted by callback during training
    """

    def __init__(self, processor, vocab_size=None, compress_ratio=8):
        self.processor = processor
        self.vocab_size = vocab_size
        self._ratio = mp.Value('d', compress_ratio)

    @property
    def compress_ratio(self) -> int | float:
        return self._ratio.value

    @compress_ratio.setter
    def compress_ratio(self, value: int | float):
        self._ratio.value = value

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        out = self.processor.apply_chat_template(
            [item["messages"] for item in batch],
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            compress_ratio=self.compress_ratio,
            return_tensors="pt",
            padding=True,
            multi_turn_reasoning=True,
        )

        # Remap OOB tokens for small vocab debug configs
        if self.vocab_size is not None and self.vocab_size < self.processor.longtext_token_id:
            pad_slot = self.vocab_size - 1
            nonpad_range = pad_slot
            for key in ("input_ids", "longtext_input_ids"):
                t = out[key]
                is_pad = t == self.processor.longtext_token_id
                t[~is_pad] = t[~is_pad] % nonpad_range
                t[is_pad] = pad_slot

        # Labels: -100 for non-assistant positions
        labels = out["input_ids"].clone()
        labels[~out["assistant_masks"].bool()] = -100

        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
            "labels": labels,
            "longtext_input_ids": out["longtext_input_ids"],
            "longtext_cu_seqlens": out["longtext_cu_seqlens"],
            "longtext_num_tokens": out["longtext_num_tokens"],
        }


class SelfDistillCollator:
    """Collator for on-policy self-distillation between LatentSeeker (student)
    and vanilla Qwen3VL (teacher).

    Produces two versions of each sample:
      - Student path: longtext → LatentSeekerEncoder → compressed latent tokens
      - Teacher path: longtext inlined as plain text (no compression)

    The teacher sees full raw text while the student sees compressed latent
    representations. Distillation loss is computed on the shared response tokens.

    Args:
        processor: LatentSeekerProcessor instance.
        student_compress_ratio: Compression ratio for the student path.
        teacher_max_length: Max sequence length for the teacher path. If the
            teacher's uncompressed sequence exceeds this, longtext is truncated
            from the left (keeping the response).
        max_length: Max sequence length for the student path.
    """

    def __init__(
        self,
        processor,
        student_compress_ratio: int | float = 8,
        teacher_max_length: int = 4096,
        max_length: int = 2048,
        student_thinking=False,
        teacher_thinking=False,
    ):
        self.processor = processor
        self.student_compress_ratio = student_compress_ratio
        self.teacher_max_length = teacher_max_length
        self.max_length = max_length
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking



    @staticmethod
    def _inline_longtext_in_messages(messages: list[dict]) -> list[dict]:
        """Replace longtext content blocks with text so the processor
        does NOT apply compression (no <|longtext_pad|> placeholders).

        Simply renames the key from "longtext" to "text" so the Jinja
        template renders it as inline text instead of a placeholder.
        """
        inlined = []
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, list):
                inlined.append(msg)
                continue

            new_content = []
            for item in content:
                if "longtext" in item:
                    # Rename key: longtext → text (template sees 'text', renders inline)
                    new_item = {k: v for k, v in item.items() if k != "longtext"}
                    longtext_val = item["longtext"]
                    if isinstance(longtext_val, list):
                        longtext_val = "\n\n".join(longtext_val)
                    new_item["text"] = longtext_val
                    new_item["type"] = "text"
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            inlined.append({**msg, "content": new_content})

        return inlined






    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        all_messages = [item["messages"] for item in batch]

        # ================================================================
        # Student path: compressed (longtext → latent tokens)
        # ================================================================
        student_out = self.processor.apply_chat_template(
            all_messages,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            compress_ratio=self.student_compress_ratio,
            return_tensors="pt",
            padding=True,
            multi_turn_reasoning=True,
        )

        student_labels = student_out["input_ids"].clone()
        student_labels[~student_out["assistant_masks"].bool()] = -100

        # ================================================================
        # Teacher path: no compression (longtext inlined as plain text)
        # ================================================================
        teacher_messages = [self._inline_longtext_in_messages(msgs) for msgs in all_messages]

        teacher_out = self.processor.apply_chat_template(
            teacher_messages,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            multi_turn_reasoning=True,
            # No compress_ratio → longtext is inlined, no latent encoding
        )

        # Truncate teacher if too long (from left, keeping response)
        teacher_input_ids = teacher_out["input_ids"]
        teacher_attention_mask = teacher_out["attention_mask"]
        teacher_assistant_mask = teacher_out["assistant_masks"]

        teacher_input_ids, teacher_attention_mask, teacher_assistant_mask = (
            self._truncate_teacher_from_left(
                teacher_input_ids,
                teacher_attention_mask,
                teacher_assistant_mask,
                self.teacher_max_length,
            )
        )

        teacher_labels = teacher_input_ids.clone()
        teacher_labels[~teacher_assistant_mask.bool()] = -100

        # Pad teacher to max_length if needed (right pad)
        teacher_seq_len = teacher_input_ids.shape[1]
        if teacher_seq_len < self.max_length:
            pad_len = self.max_length - teacher_seq_len
            pad_id = self.processor.tokenizer.pad_token_id or 0
            teacher_input_ids = torch.nn.functional.pad(
                teacher_input_ids, (0, pad_len), value=pad_id
            )
            teacher_attention_mask = torch.nn.functional.pad(
                teacher_attention_mask, (0, pad_len), value=0
            )
            teacher_labels = torch.nn.functional.pad(
                teacher_labels, (0, pad_len), value=-100
            )

        return {
            # Student inputs (compressed)
            "student_input_ids": student_out["input_ids"],
            "student_attention_mask": student_out["attention_mask"],
            "student_labels": student_labels,
            "student_longtext_input_ids": student_out.get("longtext_input_ids"),
            "student_longtext_cu_seqlens": student_out.get("longtext_cu_seqlens"),
            "student_longtext_num_tokens": student_out.get("longtext_num_tokens"),
            # Teacher inputs (uncompressed)
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_labels": teacher_labels,
        }


