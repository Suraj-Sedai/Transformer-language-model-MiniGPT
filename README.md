# MiniGPT
## Transformer Language Model from Scratch

MiniGPT is a **decoder-only, GPT-style Transformer language model** implemented **entirely from first principles in Python using PyTorch**.  
The project provides a clean, transparent, and reproducible reference implementation demonstrating how modern transformer-based language models are built and trained internally—**without relying on pretrained models or high-level language modeling libraries**.

This repository is intended for **students, researchers, and machine learning engineers** seeking a practical, end-to-end understanding of GPT-style architectures aligned with real-world engineering practices.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Design Philosophy](#design-philosophy)  
3. [System Architecture](#system-architecture)  
4. [Core Components](#core-components)  
   - [Tokenizer (Byte Pair Encoding)](#tokenizer-byte-pair-encoding)  
   - [Embedding Layer](#embedding-layer)  
   - [Transformer Blocks](#transformer-blocks)  
   - [Multi-Head Self-Attention](#multi-head-self-attention)  
   - [Feed-Forward Network](#feed-forward-network)  
5. [Model Configuration](#model-configuration)  
6. [Training Setup](#training-setup)  
7. [Training Results](#training-results)  
8. [Text Generation](#text-generation)  
9. [Checkpointing](#checkpointing)  
10. [Use Cases](#use-cases)  
11. [Future Improvements](#future-improvements)  
12. [Author](#author)  
13. [License](#license)  

---

## Project Overview

MiniGPT implements the complete autoregressive language modeling pipeline:

```
Raw Text
   ↓
BPE Tokenizer
   ↓
Token IDs
   ↓
Token + Positional Embeddings
   ↓
Transformer Blocks (× N)
   ↓
Layer Normalization
   ↓
Language Model Head
   ↓
Next-Token Logits
```

The project emphasizes:
- Architectural correctness
- Implementation clarity
- Training stability
- Full reproducibility

Rather than focusing on scale or performance tricks, MiniGPT prioritizes **understanding how GPT-style models work under the hood**.

---

## Design Philosophy

- Implemented fully **from scratch**
- Minimal abstraction leakage
- No pretrained models
- Faithful to modern GPT design conventions
- Engineering-focused rather than academic

Key design choices include:
- Pre-LayerNorm Transformer blocks
- Causal self-attention
- AdamW optimization
- Cosine learning rate scheduling
- Mixed-precision GPU training
- Robust checkpointing

---

## System Architecture

MiniGPT follows a standard **decoder-only Transformer architecture** for autoregressive language modeling.

### High-Level Flow

```
Input Tokens
   ↓
Embedding Layer
   ↓
N × Transformer Blocks
   ↓
Final LayerNorm
   ↓
Linear Language Model Head (Weight Tied)
   ↓
Vocabulary Logits
```

---

## Core Components

### Tokenizer (Byte Pair Encoding)

- Custom Byte Pair Encoding (BPE) tokenizer implemented from scratch
- Learns merge rules directly from training data
- Converts raw text to integer token IDs
- Supports both encoding and decoding

**Primary class:** `BPETokenizer`

---

### Embedding Layer

- Learnable token embeddings
- Learnable positional embeddings
- Combined and passed into the Transformer stack

---

### Transformer Blocks

Each Transformer block follows a **Pre-Layer Normalization (Pre-LN)** design:

```
x = x + MultiHeadSelfAttention(LayerNorm(x))
x = x + FeedForwardNetwork(LayerNorm(x))
```

---

### Multi-Head Self-Attention

- Scaled dot-product attention
- Multiple attention heads
- Causal masking to enforce autoregressive behavior
- Output projection back to embedding dimension

---

### Feed-Forward Network

- Two-layer position-wise MLP
- GELU activation
- Applied independently to each token position

---

## Model Configuration

### Architecture

| Component | Value |
|---------|-------|
| Model Type | Decoder-only GPT |
| Transformer Layers | 6 |
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| FFN Hidden Dimension | 2048 |
| Max Context Length | 256 |
| Dropout | 0.1 |

### Tokenizer

| Component | Value |
|---------|-------|
| Tokenizer Type | Custom BPE |
| Vocabulary Size | 20,000 |
| Merge Operations | 19,059 |

### Parameters

| Metric | Value |
|------|------|
| Total Parameters | 29,274,112 |
| Trainable Parameters | 29,274,112 |
| Model Size (fp32) | 111.67 MB |

---

## Training Setup

- **Dataset:** WikiText-2
- **Objective:** Autoregressive next-token prediction
- **Optimizer:** AdamW
- **Scheduler:** Cosine Annealing
- **Precision:** Mixed precision (fp16 on GPU)
- **Techniques:** Gradient accumulation, gradient clipping

---

## Training Results

| Metric | Value |
|------|------|
| Final Training Perplexity | 2.74 |
| Throughput | ~84,000 tokens/sec (GPU) |
| Training Duration | ~11.5 hours |

Training loss decreased smoothly and monotonically, indicating correct causal masking, stable optimization, and a valid implementation.

---

## Text Generation

MiniGPT supports **autoregressive text generation** using greedy or sampling-based decoding.

---

## Checkpointing

```
checkpoints/
├── model.pt
├── tokenizer.json
└── config.json
```

---

## Use Cases

- Educational reference for transformer architectures
- Learning GPT-style models from first principles
- Experimentation with custom transformer designs
- Foundation for more advanced language modeling projects

---

## Future Improvements

- Top-k and top-p decoding
- Longer context windows
- Larger datasets
- Optimized attention

---

## Author

**Suraj Sedai**

---

## License

Educational and research use only.
