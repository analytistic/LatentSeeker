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
import os
import re
import sys

import torch

from transformers import AutoModel, TextStreamer

from src.evaluation.metrics import Metrics
from src.models.LatentSeeker.processing_LatentSeeker import LatentSeekerProcessor

# Match optional <think>...</think> followed by optional newlines and the answer.
# Handles: well-formed, truncated (no </think>), no think tags, and cases
# where the model continues from generation prompt (no reopening <think>).
_THINK_RE = re.compile(r"^(?:<think>\n?)?(.*?)</think>\s*\n*\s*(.*)", re.DOTALL)


def parse_generation(text: str, no_think: bool = False) -> dict:
    """Split generated text into reasoning and final answer.

    When ``no_think=True``, the model skips the thinking block and the
    entire output is the answer. When ``no_think=False``, the model is
    expected to produce ``<think>...</think>`` — if ``</think>`` is
    missing, the model hit ``max_new_tokens`` mid-think (truncated).

    Returns ``{"reasoning": str, "predicted": str}``.
    """
    if no_think:
        return {"reasoning": "", "predicted": text.strip()}

    match = _THINK_RE.search(text)
    if match:
        return {
            "reasoning": match.group(1).strip(),
            "predicted": match.group(2).strip(),
        }
    # Expected </think> but not found → truncated thinking loop
    return {"reasoning": text.strip(), "predicted": ""}


@torch.no_grad()
def generate(
    model,
    processor,
    samples,
    compress_ratio: int | float,
    max_new_tokens: int,
    device: str,
    stream: bool = False,
    no_think: bool = False,
) -> tuple[list[dict], dict]:
    """Run generation and return (records, summary)."""
    records = []
    metrics = Metrics()
    n = 0
    mean_f1 = 0.0
    mean_recall = 0.0
    mean_lt = 0.0
    truncated = 0

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
            no_think=no_think,
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
        parsed = parse_generation(gen_text, no_think=no_think)
        if not no_think and parsed["predicted"] == "" and gen_text:
            truncated += 1
        lt_tokens = inputs.get("longtext_num_tokens", [])
        lt_total = sum(lt_tokens) if lt_tokens else 0

        scores = metrics.best_f1(parsed["predicted"], sample["answers"])

        records.append({
            "id": sample["id"],
            "question": sample["question"],
            "reasoning": parsed["reasoning"],
            "predicted": parsed["predicted"],
            "answers": sample["answers"],
            "longtext_tokens": lt_total,
            "recall": scores["recall"],
            "f1": scores["f1"],
        })

        # Print progress
        q = sample["question"]
        ref = sample.get("answers", "")
        print(f"\n--- Sample {i} (compress_ratio={compress_ratio}) ---")
        print(f"Q:    {q[:120]}{'...' if len(q) > 120 else ''}")
        print(f"R:    {parsed['reasoning'][:120]}{'...' if len(parsed['reasoning']) > 120 else ''}")
        print(f"A:    {parsed['predicted'][:200]}{'...' if len(parsed['predicted']) > 200 else ''}")
        print(f"Ref:  {ref[:200]}{'...' if len(ref) > 200 else ''}")
        n += 1
        mean_f1 += (scores["f1"] - mean_f1) / n
        mean_recall += (scores["recall"] - mean_recall) / n
        mean_lt += (lt_total - mean_lt) / n
        print(f"LT:   {lt_total} tokens  |  R={scores['recall']:.3f}  F1={scores['f1']:.3f}  |  avg_R={mean_recall:.4f}  avg_F1={mean_f1:.4f}")
        sys.stdout.flush()

    print(f"\n>>> compress_ratio={compress_ratio}  |  avg_lt={mean_lt:.0f}  avg_recall={mean_recall:.4f}  avg_f1={mean_f1:.4f}  ({n} samples, {truncated} truncated)")
    return records, {"avg_lt": mean_lt, "avg_recall": mean_recall, "avg_f1": mean_f1, "samples": n, "truncated": truncated}


def main():
    parser = argparse.ArgumentParser(description="QA evaluation for LatentSeeker")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", default="squad")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate on")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--compress_ratio", type=float, nargs="+", default=[1.0])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-think", action="store_true", help="Skip <think> block in generation prompt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "samples": len(samples),
        "no_think": args.no_think,
        "compress_ratios": {},
    }

    records = []
    for cr in args.compress_ratio:
        print(f"\n{'='*60}")
        print(f"Generating with compress_ratio={cr}")
        print(f"{'='*60}")
        records, cr_summary = generate(model, processor, samples, cr, args.max_new_tokens, device, stream=args.stream, no_think=args.no_think)

        cr_path = os.path.join(args.output_dir, f"cr_{cr}.jsonl")
        with open(cr_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary["compress_ratios"][str(cr)] = cr_summary

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {len(records)} predictions to {args.output_dir}/")


if __name__ == "__main__":
    main()
