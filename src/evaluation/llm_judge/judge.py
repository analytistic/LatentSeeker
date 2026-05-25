"""Judge: score / compare multi-turn QA via an API model.

Usage (library):
    from src.evaluation.llm_judge.judge import Judge

    judge = Judge(base="http://localhost:8000/v1", protocol="openai")

    # Single model scoring
    result = judge({"ls": "rollouts/ls.jsonl"})
    # → {"ls": [{"id": "0", "turns": [{"turn": 1, "score": 4, ...}, ...]}]}

    # Multi-model comparison
    result = judge({"ls": "rollouts/ls.jsonl", "qwen": "rollouts/qwen.jsonl"})
    # → {"ls": ..., "qwen": ..., "compare": [{"id": "0", "turns": [{turn, winner, ...}]}]}

Usage (standalone):
    python -m src.evaluation.llm_judge.judge \\
        --inputs ls=rollouts/ls.jsonl qwen=rollouts/qwen.jsonl \\
        --output reports/result.json
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

SCORE_PROMPT = """You are evaluating a model's answer in a conversation. Assess the final assistant response based on accuracy, completeness, and relevance to the conversation context.

Conversation:
{messages}

Score the final assistant response on a scale of 1-5:
1 = completely wrong / irrelevant
2 = partially correct but missing key points
3 = mostly correct with minor issues
4 = correct and comprehensive
5 = perfect, clear, and complete

Output ONLY a JSON object:
{{"score": <int 1-5>, "reasoning": "<one-sentence explanation>"}}"""


# ── Internal API client ───────────────────────────────────────────────


class _JudgeAPI:
    def __init__(
        self,
        base: str = "https://api.deepseek.com/anthropic",
        key: str | None = None,
        protocol: Literal["anthropic", "openai"] = "openai",
        model: str = "deepseek-v4-flash",
    ):
        self.base = base.rstrip("/")
        self.key = key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.protocol = protocol
        self.model = model

    def __call__(self, system: str, prompt: str) -> str | None:
        if self.protocol == "anthropic":
            url = f"{self.base}/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.key,
                "anthropic-version": "2023-06-01",
            }
            body = {
                "model": self.model,
                "max_tokens": 256,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }
        else:
            url = f"{self.base}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
            }
            body = {
                "model": self.model,
                "max_tokens": 256,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }

        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            print(f"  [error] HTTP {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"  [error] {e}", file=sys.stderr)
            return None

        try:
            if self.protocol == "anthropic":
                for b in result.get("content", []):
                    if b.get("type") == "text":
                        return b["text"]
            else:
                return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None


# ── Judge class ───────────────────────────────────────────────────────


class Judge:
    """Evaluate / compare multi-turn QA via an API model.

    Usage::

        judge = Judge(base="...", protocol="openai")

        # Score one model
        report = judge({"ls": "rollouts/ls.jsonl"})

        # Compare multiple models
        report = judge({"ls": "rollouts/ls.jsonl", "qwen": "rollouts/qwen.jsonl"})
    """

    def __init__(
        self,
        base: str = "https://api.deepseek.com/anthropic",
        key: str | None = None,
        protocol: Literal["anthropic", "openai"] = "openai",
        model: str = "deepseek-v4-flash",
        max_workers: int = 4,
    ):
        self.api = _JudgeAPI(base, key, protocol, model)
        self.max_workers = max_workers

    # ── Public API ────────────────────────────────────────────────────

    def __call__(self, inputs: dict[str, str | list[dict]]) -> dict:
        """Run scoring / comparison on one or more model outputs.

        Args:
            inputs: ``{name: jsonl_path_or_record_list, ...}``.
                    If single key → score only.
                    If multiple keys → score each + compare.

        Returns:
            ``{name: [...], "compare": [...]}``
        """
        data = self.check(inputs)

        # Score each model (parallel)
        result = self._score_all(data)

        # Compare if multiple models
        names = list(data.keys())
        if len(names) >= 2:
            result["compare"] = self._compare_all(data)

        return result

    def check(self, inputs: dict[str, str | list[dict]]) -> dict[str, list[dict]]:
        """Normalize inputs: path → read JSONL, list → keep as-is."""
        result = {}
        for name, value in inputs.items():
            if isinstance(value, str):
                with open(value) as f:
                    result[name] = [json.loads(line) for line in f]
            else:
                result[name] = value
        return result

    # ── Internals ─────────────────────────────────────────────────────

    def _split_turns(self, messages: list) -> list[dict]:
        """Split a multi-turn conversation into per-turn records.

        Each turn has its context (preceding messages + this turn's QA)
        and the extracted question + predicted answer.
        """
        turns = []
        ctx_so_far = []
        for msg in messages:
            ctx_so_far.append(msg)
            if msg["role"] == "assistant":
                # Extract question from the user message in this turn
                question = ""
                for m in reversed(ctx_so_far):
                    if m["role"] == "user":
                        for c in m.get("content", []):
                            if isinstance(c, dict) and c.get("type") == "text":
                                question = c["text"].strip()
                        break
                predicted = ""
                for c in msg.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        predicted = c["text"].strip()

                turns.append({
                    "turn": len(turns) + 1,
                    "question": question,
                    "predicted": predicted,
                    # Include full context up to this turn
                    "messages": list(ctx_so_far),
                })
        return turns

    def _score_one_turn(self, messages: list) -> dict:
        """Score a single turn's final assistant answer."""
        prompt = SCORE_PROMPT.format(messages=json.dumps(messages, ensure_ascii=False))
        text = self.api("You are a helpful judge.", prompt)
        if text is None:
            return {"score": 0, "reasoning": "API error"}

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"score": 0, "reasoning": text}

    def _score_one_record(self, record: dict) -> dict:
        """Score all turns in a single record."""
        turns = self._split_turns(record["messages"])
        for t in turns:
            r = self._score_one_turn(t["messages"])
            t["score"] = r.get("score", 0)
            t["reasoning"] = r.get("reasoning", "")
            # Remove redundant messages from output
            del t["messages"]
        return {"id": record["id"], "turns": turns}

    def _score_all(self, data: dict[str, list[dict]]) -> dict:
        """Score all records for all models, parallelized across records."""
        result = {}
        for name, records in data.items():
            scored = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self._score_one_record, r): r["id"] for r in records}
                for future in as_completed(futures):
                    scored.append(future.result())

            # Restore original order
            id_order = [r["id"] for r in records]
            id_idx = {r["id"]: i for i, r in enumerate(records)}
            scored.sort(key=lambda s: id_idx.get(s["id"], 0))

            result[name] = scored
        return result

    def _compare_all(self, data: dict[str, list[dict]]) -> list[dict]:
        """Cross-model comparison: align by id, compare turn by turn."""
        # Index each model by id
        indexed: dict[str, dict[str, dict]] = {}
        for name, records in data.items():
            indexed[name] = {r["id"]: r for r in records}

        all_ids = sorted(set.union(*[set(m.keys()) for m in indexed.values()]))
        names = list(data.keys())

        results = []
        for sid in all_ids:
            # Split turns for all models for this id
            model_turns: dict[str, list[dict]] = {}
            for name in names:
                if sid in indexed.get(name, {}):
                    model_turns[name] = self._split_turns(indexed[name][sid]["messages"])

            if not model_turns:
                continue

            max_turns = max(len(t) for t in model_turns.values())
            turn_results = []

            for turn_idx in range(max_turns):
                scores = {}
                for name, turns in model_turns.items():
                    if turn_idx < len(turns):
                        r = self._score_one_turn(turns[turn_idx]["messages"])
                        scores[name] = {
                            "score": r.get("score", 0),
                            "reasoning": r.get("reasoning", ""),
                            "predicted": turns[turn_idx]["predicted"],
                            "question": turns[turn_idx]["question"],
                        }

                # Determine winner(s)
                max_score = max(s["score"] for s in scores.values())
                winners = [n for n, s in scores.items() if s["score"] == max_score]
                winner = winners[0] if len(winners) == 1 else "tie"

                turn_results.append({
                    "turn": turn_idx + 1,
                    "question": next((s["question"] for s in scores.values() if s["question"]), ""),
                    "scores": {n: s["score"] for n, s in scores.items()},
                    "predicted": {n: s["predicted"] for n, s in scores.items()},
                    "winner": winner,
                })

            results.append({"id": sid, "turns": turn_results})

        return results


# ── Standalone CLI ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Judge: score / compare QA")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="name=path pairs, e.g. ls=rollouts/ls.jsonl qwen=rollouts/qwen.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-base", default="https://api.deepseek.com/anthropic")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-protocol", default="openai", choices=["anthropic", "openai"])
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    inputs = {}
    for item in args.inputs:
        name, _, path = item.partition("=")
        inputs[name] = path

    judge = Judge(
        base=args.api_base,
        key=args.api_key,
        protocol=args.api_protocol,
        model=args.judge_model,
        max_workers=args.max_workers,
    )

    result = judge(inputs)
    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
