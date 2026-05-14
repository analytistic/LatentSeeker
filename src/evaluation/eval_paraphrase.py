"""Paraphrase reconstruction evaluation for LatentSeeker.

Usage:
    python -m src.evaluation.eval_paraphrase \
        --model_path outputs/stage1/checkpoint-100 \
        --data_path data/debug/processed_debug \
        --compress_ratio 1 2 4
"""

import argparse
import sys

import torch

from transformers import AutoModel, TextStreamer

from src.dataset.get_wiki import get_wiki
from src.evaluation.metrics import Metrics
from src.models.LatentSeeker.processing_LatentSeeker import LatentSeekerProcessor


def _get_ref_text(sample) -> str:
    """Extract reference text from assistant message."""
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            for c in msg["content"]:
                if c.get("type") == "text":
                    return c["text"].strip()
    return ""


@torch.no_grad()
def evaluate(
    model,
    processor,
    samples,
    compress_ratio: int | float,
    max_new_tokens: int,
    metrics: Metrics,
    stream: bool = False,
) -> dict:
    """Run paraphrase evaluation at a given ``compress_ratio``."""
    hypotheses = []
    references = []
    gen_ids_list = []
    ref_ids_list = []

    for i, sample in enumerate(samples):
        messages = [m for m in sample["messages"] if m["role"] != "assistant"]
        ref_text = _get_ref_text(sample)

        # --- Tokenize ---
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            no_think=True,
            return_dict=True,
            return_tensors="pt",
            compress_ratio=compress_ratio,
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Remap OOB tokens for small vocab debug configs
        vocab_size = model.config.text_config.vocab_size
        if vocab_size is not None:
            for key in ("input_ids", "longtext_input_ids"):
                if key in inputs:
                    t = inputs[key]
                    if isinstance(t, torch.Tensor) and t.numel() > 0 and t.max().item() >= vocab_size:
                        longtext_token_id = processor.longtext_token_id
                        pad_slot = vocab_size - 1
                        is_pad = t == longtext_token_id
                        t[~is_pad] = t[~is_pad] % pad_slot
                        t[is_pad] = pad_slot

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
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path} ...")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    ).eval()

    processor = LatentSeekerProcessor.from_pretrained(args.model_path)
    metrics = Metrics(tokenizer=processor.tokenizer)

    print(f"Loading data from {args.data_path} ...")
    samples = get_wiki(args.data_path, max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"\n{'='*60}")
    print(f"{'compress_ratio':>14}  {'token_acc':>10}  {'BLEU':>8}")
    print(f"{'='*60}")

    for cr in args.compress_ratio:
        result = evaluate(
            model, processor, samples, cr, args.max_new_tokens, metrics,
            stream=args.stream,
        )
        print(f"{'='*60}")
        print(f"{cr:>14.1f}  {result['token_accuracy']:>10.4f}  {result['bleu']:>8.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
