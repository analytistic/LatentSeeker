"""QA evaluation for LatentSeeker.

Generates answers for a processed QA dataset and saves predictions to JSONL.

Usage:
    python -m src.evaluation.eval_qa \
        --model_path outputs/pretrain/stage1/checkpoint-500 \
        --data_path data/squad/processed_squad \
        --output outputs/eval/squad_preds.jsonl \
        --compress_ratio 1 2 4
"""

import argparse
import importlib
import json
import sys

import torch

from transformers import AutoModel, TextStreamer

from src.models.LatentSeeker.processing_LatentSeeker import LatentSeekerProcessor


@torch.no_grad()
def generate(
    model,
    processor,
    samples,
    compress_ratio: int | float,
    max_new_tokens: int,
    device: str,
    stream: bool = False,
) -> list[dict]:
    """Run generation and return prediction records."""
    records = []

    for i, sample in enumerate(samples):
        messages = sample["messages"]

        # --- Tokenize ---
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            compress_ratio=compress_ratio,
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Remap OOB tokens for debug configs
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
        gen_text = processor.decode(gen_ids, skip_special_tokens=True).strip()

        records.append({
            "id": sample["id"],
            "predicted": gen_text,
            "answers": sample["answers"],
        })

        # Print progress
        print(f"\n--- Sample {i} (compress_ratio={compress_ratio}) ---")
        print(f"Q:  {sample['question'][:100]}{'...' if len(sample['question']) > 100 else ''}")
        print(f"A:  {gen_text[:200]}")
        sys.stdout.flush()

    return records


def main():
    parser = argparse.ArgumentParser(description="QA evaluation for LatentSeeker")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", default="squad")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate on")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compress_ratio", type=float, nargs="+", default=[1.0])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
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

    print(f"Loading {args.split} split from {args.data_path} ...")
    getter = importlib.import_module(f"src.dataset.get_{args.dataset}")
    samples = getter.load(args.data_path, split=args.split, max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples from split '{args.split}'")

    all_records = {}
    for cr in args.compress_ratio:
        print(f"\n{'='*60}")
        print(f"Generating with compress_ratio={cr}")
        print(f"{'='*60}")
        records = generate(model, processor, samples, cr, args.max_new_tokens, device, stream=args.stream)
        all_records[str(cr)] = records

    # Write output JSONL: each line is {id, predicted, answers}
    with open(args.output, "w") as f:
        for records in all_records.values():
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {sum(len(v) for v in all_records.values())} predictions to {args.output}")


if __name__ == "__main__":
    main()
