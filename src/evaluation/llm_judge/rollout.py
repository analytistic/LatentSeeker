"""Rollout: generate multi-turn answers from a LatentSeeker model.

Generates all turns for each sample, then writes one JSONL line per sample
with the complete conversation (all Q&A pairs).

Usage:
    python -m src.evaluation.llm_judge.rollout \
        --model_path outputs/multitask_sft \
        --data_path data/eval/multiturn.jsonl \
        --output rollouts/ls.jsonl \
        --compress_ratio 5
"""

import argparse
import json
import sys

import torch
from transformers import AutoModel

from src.models.LatentSeeker.processing_LatentSeeker import LatentSeekerProcessor


def _build_ctx(original_messages: list, turn: int, model_answers: list[str]) -> list:
    """Build context up to the given turn, using model's own previous answers."""
    ctx = []
    asst_count = 0
    for msg in original_messages:
        if msg["role"] == "assistant":
            asst_count += 1
            if asst_count >= turn:
                break
            ctx.append({
                "role": "assistant",
                "content": [{"type": "text", "text": model_answers[asst_count - 1]}],
            })
            continue
        ctx.append(msg)
        if msg["role"] == "user" and asst_count == turn - 1:
            break
    return ctx


@torch.no_grad()
def rollout(
    model,
    processor,
    samples: list[dict],
    output_path: str,
    compress_ratio: float,
    max_new_tokens: int,
    device: str,
):
    """Generate all turns for each sample, write one line per sample."""
    fout = open(output_path, "a")

    for sample in samples:
        messages = sample["messages"]
        sid = sample.get("id", str(sample.get("_id", "")))
        num_turns = sum(1 for m in messages if m["role"] == "assistant")

        model_answers: list[str] = []
        full_msgs = []

        for turn in range(1, num_turns + 1):
            ctx = _build_ctx(messages, turn, model_answers)

            # Tokenize + generate
            inputs = processor.apply_chat_template(
                ctx,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                compress_ratio=compress_ratio,
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            vocab_size = model.config.text_config.vocab_size
            if vocab_size is not None:
                for key in ("input_ids", "longtext_input_ids"):
                    if key in inputs:
                        t = inputs[key]
                        if isinstance(t, torch.Tensor) and t.numel() > 0 and t.max().item() >= vocab_size:
                            ltid = processor.longtext_token_id
                            pad = vocab_size - 1
                            is_pad = t == ltid
                            t[~is_pad] = t[~is_pad] % pad
                            t[is_pad] = pad

            prompt_len = inputs["input_ids"].shape[-1]
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            gen_ids = output_ids[0, prompt_len:].tolist()
            predicted = processor.decode(gen_ids, skip_special_tokens=True).strip()
            model_answers.append(predicted)

            # Append this turn to the running full messages
            full_msgs.extend(ctx[len(full_msgs):])
            full_msgs.append({
                "role": "assistant",
                "content": [{"type": "text", "text": predicted}],
            })

            print(f"[{sid}] turn {turn}/{num_turns}: {predicted[:80]}...")
            sys.stdout.flush()

        # One line per sample: complete multi-turn conversation
        record = {"id": sid, "messages": full_msgs}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-turn QA rollout")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compress_ratio", type=float, default=5.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path} ...")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    ).eval()

    processor = LatentSeekerProcessor.from_pretrained(args.model_path)

    samples = []
    with open(args.data_path) as f:
        for i, line in enumerate(f):
            if args.max_samples and i >= args.max_samples:
                break
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples")

    rollout(model, processor, samples, args.output, args.compress_ratio, args.max_new_tokens, device)
    print(f"Done — results written to {args.output}")


if __name__ == "__main__":
    main()
