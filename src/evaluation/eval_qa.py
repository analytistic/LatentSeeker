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
import threading
import time
from collections import deque

import torch

from transformers import AutoModel, AutoProcessor, TextStreamer

from src.evaluation.metrics import Metrics

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


def _build_messages(sample: dict, baseline: bool) -> list:
    """Prepare messages: convert longtext to plain text for baseline."""
    messages = sample["messages"]
    if not baseline:
        return messages
    return [
        {
            "role": m["role"],
            "content": [
                {"type": "text", "text": c.get("longtext") or c.get("text", "")}
                if c.get("type") in ("longtext", "text") else c
                for c in m.get("content", [])
            ],
        }
        for m in messages
    ]


@torch.no_grad()
def infer_one(
    model, processor, sample, compress_ratio, max_new_tokens, device, stream, no_think, baseline
) -> dict:
    """Generate answer for a single sample. Returns raw record (no metrics)."""
    messages = _build_messages(sample, baseline)

    # --- Tokenize ---
    tt_kwargs = dict(
        messages=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        no_think=no_think,
    )
    if not baseline:
        tt_kwargs["compress_ratio"] = compress_ratio
    inputs = processor.apply_chat_template(**tt_kwargs)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Remap OOB tokens for debug configs
    vocab_size = model.config.text_config.vocab_size
    if vocab_size is not None:
        for key in ("input_ids", "longtext_input_ids"):
            if key in inputs:
                t = inputs[key]
                if isinstance(t, torch.Tensor) and t.numel() > 0 and t.max().item() >= vocab_size:
                    longtext_token_id = getattr(processor, 'longtext_token_id', None)
                    if longtext_token_id is not None:
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

    n_reasoning = 0
    if not no_think and parsed["predicted"] == "" and gen_text:
        truncated = True
    elif not no_think and parsed["reasoning"]:
        n_reasoning = len(processor.tokenizer.encode(parsed["reasoning"]))
        truncated = False
    else:
        truncated = False

    return {
        "id": sample["id"],
        "question": sample["question"],
        "reasoning": parsed["reasoning"],
        "n_reasoning": n_reasoning,
        "predicted": parsed["predicted"],
        "answers": sample["answers"],
        "longtext": 0 if baseline else len(inputs.get("longtext_input_ids", [])),
        "n_latent": 0 if baseline else sum(inputs.get("longtext_num_tokens", []) or []),
        "truncated": truncated,
    }


def infer_thread(
    samples, buffer, lock, done,
    model, processor, compress_ratio, max_new_tokens, device, stream, no_think, baseline,
):
    """Producer: infer each sample, push raw record to buffer."""
    done.clear()
    for i, sample in enumerate(samples):
        record = infer_one(model, processor, sample, compress_ratio, max_new_tokens, device, stream, no_think, baseline)
        with lock:
            buffer.append(record)

        # Print progress
        q = sample["question"]
        ref = sample.get("answers", "")
        print(f"\n\n--- Sample {i} (compress_ratio={compress_ratio}) ---")
        print(f"Q:    {q[:120]}{'...' if len(q) > 120 else ''}")
        print(f"R:    {record['reasoning'][:120]}{'...' if len(record['reasoning']) > 120 else ''}")
        print(f"A:    {record['predicted'][:200]}{'...' if len(record['predicted']) > 200 else ''}")
        print(f"Ref:  {ref[:200]}{'...' if len(ref) > 200 else ''}")
        sys.stdout.flush()
    done.set()


def consume_thread(
    buffer, lock, done, output_path, summary_path, interval, compress_ratio,
):
    """Consumer: poll buffer, write JSONL, compute metrics, update summary."""
    metrics = Metrics()
    n = 0
    mean_f1 = 0.0
    mean_recall = 0.0
    mean_latent = 0.0
    mean_longtext = 0.0
    truncated = 0
    mean_reasoning_len = 0.0
    reasoning_n = 0
    summary = {}

    while True:
        # Drain buffer in batch
        batch = []
        with lock:
            while buffer:
                batch.append(buffer.popleft())

        if not batch:
            if done.is_set():
                break
            time.sleep(interval)
            continue

        # Compute all metrics, then write batch atomically
        for record in batch:
            scores = metrics.best_f1(record["predicted"], record["answers"])
            record["f1"] = scores["f1"]
            record["recall"] = scores["recall"]

            n += 1
            mean_f1 += (scores["f1"] - mean_f1) / n
            mean_recall += (scores["recall"] - mean_recall) / n
            mean_latent += (record["n_latent"] - mean_latent) / n
            mean_longtext += (record["longtext"] - mean_longtext) / n
            if record["truncated"]:
                truncated += 1
            if record["n_reasoning"]:
                reasoning_n += 1
                mean_reasoning_len += (record["n_reasoning"] - mean_reasoning_len) / reasoning_n

            print(f"LT:   {record['longtext']}→{record['n_latent']}  |  reasoning={record['n_reasoning']}tok  |  R={record['recall']:.3f}  F1={record['f1']:.3f}  |  avg_R={mean_recall:.4f}  avg_F1={mean_f1:.4f}\n")
            sys.stdout.flush()

        with open(output_path, "a") as f:
            for record in batch:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = {
            "longtext": mean_longtext, "latent": mean_latent,
            "reasoning_tok": mean_reasoning_len,
            "recall": mean_recall, "f1": mean_f1,
            "samples": n, "truncated": truncated,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n\n>>> compress_ratio={compress_ratio}  |  longtext={mean_longtext:.0f}→latent={mean_latent:.0f}  reasoning={mean_reasoning_len:.0f}tok  R={mean_recall:.4f}  F1={mean_f1:.4f}  ({n} samples, {truncated} truncated)\n")

    return summary


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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-think", action="store_true", help="Skip <think> block in generation prompt")
    parser.add_argument("--baseline", action="store_true", help="Baseline mode (no longtext compression)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path} ...")
    model = AutoModel.from_pretrained(
        args.model_path,
        device_map=args.device,
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading {args.split} split from {args.data_path} ...")
    getter = importlib.import_module(f"src.dataset.get_{args.dataset}")
    samples = getter.load(args.data_path, split=args.split, max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples from split '{args.split}'")

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "samples": len(samples),
        "baseline": args.baseline,
        "no_think": args.no_think,
        "compress_ratios": {},
    }

    for cr in args.compress_ratio:
        print(f"\n{'='*60}")
        print(f"Generating with compress_ratio={cr}")
        print(f"{'='*60}")

        cr_path = os.path.join(args.output_dir, f"cr_{cr}.jsonl")
        summary_path = os.path.join(args.output_dir, f"cr_{cr}_summary.json")

        buffer: deque = deque()
        lock = threading.Lock()
        done = threading.Event()

        t_infer = threading.Thread(
            target=infer_thread,
            args=(samples, buffer, lock, done,
                  model, processor, cr, args.max_new_tokens, args.device,
                  args.stream, args.no_think, args.baseline),
        )
        t_consume = threading.Thread(
            target=consume_thread,
            args=(buffer, lock, done, cr_path, summary_path, 1, cr),
        )

        t_infer.start()
        t_consume.start()
        t_infer.join()
        t_consume.join()

        with open(summary_path) as f:
            cr_summary = json.load(f)
        summary["compress_ratios"][str(cr)] = cr_summary

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved predictions to {args.output_dir}/\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
