"""Paraphrase reconstruction evaluation for LatentSeeker.

Usage:
    python scripts/eval_paraphrase.py \
        --model_path outputs/stage1/checkpoint-100 \
        --data_path data/debug/debug.jsonl \
        --compress_ratio 1

    # Multiple compression ratios for a rate–accuracy curve
    python scripts/eval_paraphrase.py \
        --model_path outputs/stage1/checkpoint-100 \
        --data_path data/debug/debug.jsonl \
        --compress_ratio 1 2 4 8
"""

import argparse
import json
import sys

import torch

from transformers import TextStreamer

from src.evaluation.metrics import Metrics
from src.models.LatentSeeker.modeling_LatentSeeker import (
    LatentSeekerForConditionalGeneration,
)
from src.models.LatentSeeker.processing_LatentSeeker import LatentSeekerProcessor


def load_data(path: str, max_samples: int | None = None) -> list[dict]:
    """Load jsonl lines with a ``text`` field."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


@torch.no_grad()
def evaluate(
    model,
    processor,
    samples: list[dict],
    compress_ratio: int | float,
    max_new_tokens: int,
    metrics: Metrics,
    device: str,
    stream: bool = False,
) -> dict:
    """Run paraphrase evaluation at a given ``compress_ratio``."""
    hypotheses = []
    references = []
    gen_ids_list = []
    ref_ids_list = []

    for i, sample in enumerate(samples):
        text = sample["text"].strip()

        # --- Format messages ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "longtext", "longtext": text},
                    {"type": "text", "text": "Please repeat the longtext above."},
                ],
            },
        ]

        # --- Tokenize ---
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            longtext_kwargs={"compress_ratio": compress_ratio},
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # --- Generate ---
        prompt_len = inputs["input_ids"].shape[-1]
        streamer = TextStreamer(processor.tokenizer, skip_prompt=True) if stream else None
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            streamer=streamer,
        )
        gen_ids = output_ids[0, prompt_len:].tolist()

        # --- Decode ---
        gen_text = processor.decode(gen_ids, skip_special_tokens=True).strip()
        ref_text = text

        # --- Reference BPE ids (for token_accuracy) ---
        ref_ids = processor.tokenizer.encode(ref_text, add_special_tokens=False)

        hypotheses.append(gen_text)
        references.append(ref_text)
        gen_ids_list.append(gen_ids)
        ref_ids_list.append(ref_ids)

        # Print each sample
        print(f"\n--- Sample {i} (compress_ratio={compress_ratio}) ---")
        print(f"REF:   {ref_text[:200]}{'...' if len(ref_text)>200 else ''}")
        print(f"GEN:   {gen_text[:200]}{'...' if len(gen_text)>200 else ''}")
        print(f"gen_len={len(gen_ids)}, ref_len={len(ref_ids)}")
        sys.stdout.flush()

    # --- Aggregate metrics ---
    acc_result = metrics.batch_token_accuracy(gen_ids_list, ref_ids_list)
    bleu_result = metrics.batch_bleu(hypotheses, references)

    return {
        "num_samples": len(samples),
        "compress_ratio": compress_ratio,
        "token_accuracy": acc_result["accuracy"],
        "bleu": bleu_result["bleu"],
        "per_sample": [
            {"ta": a["accuracy"], "bleu": b}
            for a, b in zip(acc_result["per_sample"], bleu_result["per_sample"])
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Paraphrase reconstruction evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--compress_ratio", type=float, nargs="+", default=[1.0])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stream", action="store_true", help="Show token-by-token generation")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path} ...")
    model = LatentSeekerForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    ).to(device).eval()

    processor = LatentSeekerProcessor.from_pretrained(args.model_path)
    metrics = Metrics(tokenizer=processor.tokenizer)

    samples = load_data(args.data_path, args.max_samples)
    print(f"Loaded {len(samples)} samples from {args.data_path}")

    print(f"\n{'='*60}")
    print(f"{'compress_ratio':>14}  {'token_acc':>10}  {'BLEU':>8}")
    print(f"{'='*60}")

    for cr in args.compress_ratio:
        result = evaluate(
            model, processor, samples, cr, args.max_new_tokens, metrics, device,
            stream=args.stream,
        )
        print(f"{'='*60}")
        print(f"{cr:>14.1f}  {result['token_accuracy']:>10.4f}  {result['bleu']:>8.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
