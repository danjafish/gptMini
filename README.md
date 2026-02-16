# gptMini

A small, minimal GPT-like **decoder-only Transformer** implementation for learning.  
Trains a next-token language model and supports autoregressive generation with optional **KV-cache**.

---

## Features

- Decoder-only Transformer blocks (pre-norm)
- Multi-Head Attention + custom scaled dot-product attention (SDPA)
- Positional encodings:
  - Learned absolute
  - Sinusoidal/Fourier
  - RoPE (rotary embeddings)
- Next-token training (teacher forcing via shifted targets)
- Generation:
  - Greedy decoding
  - Top-k sampling + temperature
  - KV-cache for faster autoregressive decoding
- Simple training/eval loop + loss plotting
- Checkpoint save/load (model + optimizer + scheduler)

---

## Dependencies

```bash
pip install torch transformers datasets matplotlib
```

---

## Dataset

WikiText-103 from Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
train = ds["train"]
val = ds["validation"]
```

Tokenizer: GPT-2 tokenizer via `transformers.AutoTokenizer`.

---

## Training objective

Next-token prediction:

- Inputs: `input_ids[:, :-1]`
- Targets: `input_ids[:, 1:]`
- Loss: cross-entropy (optionally with label smoothing)

Typical “GPT-style” dataloader:
- Concatenate tokenized documents into one long token stream (insert EOS between docs)
- Slice fixed-length `block_size` chunks

---

## Generation

Autoregressive decoding:
- Start from a prompt (`prompt_ids`)
- Repeatedly predict the next token from the last-step logits
- Stop on EOS or `max_steps`

KV-cache stores per-layer `(K, V)` to avoid recomputing attention over the full prefix each step.

---

## Checkpointing

Save:

```python
save_checkpoint(model, optim, "gptmini_ckpt.pt", scheduler=scheduler, extra={"epoch": epoch})
```

Load:

```python
extra = load_checkpoint(model, "gptmini_ckpt.pt", optimizer=optim, scheduler=scheduler, map_location="cuda")
model.to("cuda")
```

---

