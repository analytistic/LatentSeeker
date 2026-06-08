# Data Generation Design

## Overview

We generate multi-turn QA training data for LatentSeeker, a model that compresses
long context into latent tokens. The key challenge is creating data where:

1. **Longtext compression matters** — questions require understanding document content
2. **Cross-document reasoning** — model must synthesize across multiple documents
3. **Progressive accumulation** — documents are introduced group by group
4. **Positional referencing** — model learns to locate information by document/section
5. **Multi-step reasoning** — questions require inference, not just lookup

## Scripts

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `gen_qa.py` | `{"text": "..."}` each line | Single-longtext multi-turn QA | Baseline: one document, multiple Q&A |
| `gen_qa_multi_text.py` | `{"text": "..."}` each line | Multi-doc single-round QA | Multiple docs upfront, multiple Q&A |
| `gen_qa_multi_turn.py` | `{"text": "..."}` each line | Multi-turn multi-doc QA | Docs streamed, grouped, accumulated |

## Question Types

### Document-grounded (source=longtext)

- **summary** — Synthesize main themes across the document
- **detail** — Specific fact, definition, number, or claim
- **needle** — Find a specific piece of hidden information
- **multi_hop** — Combine info from multiple document sections
- **comparison** — Compare/contrast concepts or viewpoints
- **temporal** — Sequence, chronology, causal relationships
- **math_reasoning** — Multi-step quantitative reasoning based on numerical data in docs

### History-grounded (source=history)

- **follow_up** — Build on the previous Q&A turn (elaboration, clarification, deeper dive)
- **evolve** — Rewrite the previous question to be harder (depth, breadth, constraint, backward)

### Mixed (source=both)

- **synthesis** — Combine document info with conversation history

## Evolve Mechanism

When `evolve` is sampled, it takes the previous Q&A and makes it harder:

| Evolve Type | Operation | Example |
|-------------|-----------|---------|
| **depth** | Add reasoning steps | One-step → multi-step calculation |
| **breadth** | Combine with another document concept | Rate → rate × time × additional factor |
| **constraint** | Add edge cases or conditions | "If growth slows after year 3..." |
| **backward** | Reverse direction | Given answer, find the cause |

The evolve pipeline:
1. Pick 1–2 random seed turns from history
2. Randomly choose an evolve type
3. Call API → generate harder Q&A
4. Quality check: "Is this meaningful, solvable, non-trivial?"
5. If check fails, retry up to 3×; if all fail, skip evolve

## Token Budget

- `--max-qa-tokens` controls the total token budget for Q&A turns
- Longtext content is compressed by the processor, not counted in budget
- Token counting uses `tokenizer.apply_chat_template` on candidate messages
- When budget exceeded, the last turn is kept but generation stops

## Group-based Streaming (multi_turn)

Documents are read one-by-one from the input stream and accumulated into
variable-sized groups:

```
Stream: [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, ...]
                                ↓  group by random 1..max_group_size
Group 1: [doc1, doc2, doc3]  →  Q1, Q2, ...
Group 2: [doc4, doc5]        →  Q3, ...
Group 3: [doc6]              →  Q4, Q5, ...
   ⋮
```

Each group's documents are placed in the user message as `longtext` blocks.
Within a group, subsequent turns are text-only (no repeated longtext).
Historical groups' Q&A are shown in the prompt for cross-document reasoning.

## Document Referencing

The prompt instructs the model to reference documents by position:
- "the first document", "Document 2", "Section 2 of Document 1"
- Avoid vague references like "the document above"

This teaches the model to locate information within the compressed latent
representation during training.

## Resume & Multi-threading

- State file saves `samples_done` and `lines_read` for crash recovery
- Single-thread: pure streaming, minimal memory
- Multi-thread: pre-batch documents into conversations, parallelize via
  ThreadPoolExecutor
