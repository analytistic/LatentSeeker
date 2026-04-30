# LatentSeeker

Compress long text contexts into compact latent tokens, analogous to how vision-language models (like Qwen3-VL) compress images.

## Motivation

Long-context LLMs suffer from quadratic attention cost. LatentSeeker treats long documents the way VL models treat images — **compress first, process later**:

1. Long documents → **encoder** → compact latent tokens (e.g., 32 per doc)
2. Latent tokens replace `<|longtext_pad|>` placeholders in the text sequence
3. Text decoder runs on the compressed sequence → efficient generation

## Architecture

```
Input text: "Tell me about <|longtext_pad|>×N"
                          ↓
Long doc ─→ LongtextEncoder ─→ latent vectors (N tokens)
                          ↓
           masked_scatter replaces placeholders
                          ↓
              Qwen3VLTextModel (decoder)
                          ↓
                       lm_head
```

### LongtextEncoder

- **Flat concat + cu_seqlens**: All documents concatenated into one flat sequence, FA2 varlen attention with cumulative segment boundaries for document isolation
- **Bidirectional attention** (`LongtextEncoderBlock`): Pre-norm RMSNorm + GQA attention + MLP, independent per-segment via cu_seqlens
- **RoPE**: 1D rotary position encoding, positions reset per document segment
- **Segment pooling** (`LongertextMerger`): Each doc's tokens divided evenly into `num_queries` blocks, each block average-pooled to 1 token → MLP bridge to decoder hidden space

### Generation flow

```python
messages = [
    {"role": "user", "content": [
        {"type": "longtext", "longtext": "War and Peace full text..."},
        {"type": "text", "text": "Summarize the main themes."},
    ]},
]

# Processor handles template, tokenization, placeholders
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")

# Model forward: encode longtext → replace placeholders → generate
outputs = model.generate(**inputs)
```

## Quick start

```python
from transformers import AutoProcessor
from src.models.LatentSeeker import LatentSeekerForConditionalGeneration

processor = AutoProcessor.from_pretrained("path/to/latentseeker")
model = LatentSeekerForConditionalGeneration.from_pretrained("path/to/latentseeker")

messages = [
    {"role": "user", "content": [
        {"type": "longtext", "longtext": "Long document content here..."},
        {"type": "text", "text": "What is the key argument?"},
    ]},
]

inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(**inputs)
print(processor.decode(outputs[0]))
```


## Components

| Module                                   | Description                                                      |
| ---------------------------------------- | ---------------------------------------------------------------- |
| `LatentSeekerEncoderModel`             | Longtext encoder: embed → bidirectional blocks → merge         |
|                                          |                                                                  |
| `LatentSeekerModel`                    | Encoder + Qwen3VLTextModel                                       |
| `LatentSeekerForConditionalGeneration` | Full model with lm_head, GenerationMixin                         |
| `LatentSeekerProcessor`                | Chat template, longtext placeholder insertion, assistant masking |

## Dependencies

- Python >= 3.10
- PyTorch >= 2.10
- transformers >= 5.3.0
- datasets

## Design references

- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B) — vision model pattern (flat concat + cu_seqlens, deepstack)
- [DeepStack](https://arxiv.org/abs/2406.04334) — multi-layer feature injection from encoder to decoder
