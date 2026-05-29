"""Conversation prefix tree: align multi-turn conversations across models.

Builds a tree keyed by user message content, with assistant branches
per model. Supports tool-call scenarios where user/assistant roles
alternate. User nodes are indexed by content for O(1) lookup.

Structure::

    User("Q1")
        ├── Asst("ls", "A1") ──→ User("Q2") ──→ Asst("ls", "A2") → None
        └── Asst("qwen", "A1") ──→ User("Q2") ──→ Asst("qwen", "A2") → None
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AssistantNode:
    """One model's assistant reply. Links to the next user question."""
    model: str
    content: str
    reasoning: str = ""
    next_user: UserNode | None = None


@dataclass
class UserNode:
    """A user question shared across models. Indexed by content."""
    content: str
    assistants: dict[str, AssistantNode] = field(default_factory=dict)


class ConversationTree:
    """Prefix tree for aligning multi-turn conversations by user content.

    Usage::

        tree = ConversationTree()
        # Insert each model's conversation into the tree
        for model, records in data.items():
            for record in records:
                tree.add(record["messages"], model)

        # Aligned pairs
        for pair in tree.pairs():
            print(pair["ls"]["content"], pair["qwen"]["content"])
    """

    def __init__(self):
        self._user_index: dict[str, UserNode] = {}

    def add(self, messages: list[dict], model: str) -> AssistantNode | None:
        """Insert a conversation into the tree.

        Returns the final AssistantNode for chaining, or None if empty.
        """
        cur: UserNode | None = None
        last_asst: AssistantNode | None = None

        for msg in messages:
            if msg["role"] == "user":
                c = (msg.get("content") or "").strip()
                if cur is not None and c == cur.content:
                    continue  # skip consecutive user turns (tool response)
                nxt = self._user_index.get(c)
                if nxt is None:
                    nxt = UserNode(content=c)
                    self._user_index[c] = nxt
                    if cur is not None:
                        for a in cur.assistants.values():
                            a.next_user = nxt
                cur = nxt

            elif msg["role"] == "assistant" and cur is not None:
                c = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning_content") or "").strip()
                node = AssistantNode(model=model, content=c, reasoning=reasoning)
                cur.assistants[model] = node
                last_asst = node

        return last_asst

    def models(self) -> set[str]:
        """Return all model names present in the tree."""
        models: set[str] = set()
        for node in self._user_index.values():
            models.update(node.assistants.keys())
        return models

    def pairs(self) -> list[dict[str, str]]:
        """Yield aligned (model → assistant content) pairs per user turn.

        Only yields turns where all models have an answer. Example::

            [{"ls": "answer from ls", "qwen": "answer from qwen"}, ...]
        """
        models = sorted(self.models())
        results: list[dict[str, str]] = []
        # Walk root user nodes in insertion order
        # Roots: user nodes whose content is NOT referenced by any assistant
        refed = set()
        for node in self._user_index.values():
            for a in node.assistants.values():
                if a.next_user is not None:
                    refed.add(a.next_user.content)
        roots = [n for n in self._user_index.values() if n.content not in refed]

        # BFS from roots
        queue = list(roots)
        seen = {n.content for n in queue}
        while queue:
            node = queue.pop(0)
            if len(node.assistants) == len(models):
                entry = {}
                for m in models:
                    a = node.assistants.get(m)
                    entry[m] = a.content if a else ""
                results.append(entry)
            for a in node.assistants.values():
                if a.next_user is not None and a.next_user.content not in seen:
                    seen.add(a.next_user.content)
                    queue.append(a.next_user)

        return results

    def to_dict(self) -> dict:
        """Serialize for debugging."""
        return {
            "roots": [self._node_to_dict(n) for n in self._user_index.values()
                      if not any(a.next_user == n for u in self._user_index.values()
                                 for a in u.assistants.values())],
        }

    @staticmethod
    def _node_to_dict(n: UserNode) -> dict:
        return {
            "user": n.content[:120],
            "assistants": {
                m: {
                    "content": a.content[:120],
                    "next_user": a.next_user.content[:120] if a.next_user else None,
                }
                for m, a in n.assistants.items()
            },
        }
