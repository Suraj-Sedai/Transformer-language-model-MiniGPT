# MiniGPT – A Minimal Transformer Language Model (From Scratch)

MiniGPT is a fully functional transformer-based language model implemented entirely from scratch in Python. The project is designed as an educational and reference implementation that demonstrates the internal mechanics of modern transformer language models (GPT-style architectures) without relying on high-level deep learning frameworks.

This repository provides a transparent, step-by-step construction of a transformer language model, making it suitable for students, researchers, and engineers seeking a deep conceptual and practical understanding of transformer architectures.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Detailed Components](#detailed-components)
  - [Tokenizer (BPE)](#tokenizer-bpe)
  - [Embedding Layer](#embedding-layer)
  - [Multi-Head Self-Attention](#multi-head-self-attention)
  - [Feed-Forward Network](#feed-forward-network)
  - [Layer Normalization and Residual Connections](#layer-normalization-and-residual-connections)
- [Transformer Block](#transformer-block)
- [Training Procedure](#training-procedure)
- [Text Generation](#text-generation)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Implementation Steps](#implementation-steps)
- [Future Improvements](#future-improvements)
- [Use Cases](#use-cases)
- [License](#license)

---

## Overview

MiniGPT is a minimal yet complete implementation of a transformer-based language model. It includes all core architectural components found in modern large language models, implemented in a clear and modular fashion.

### Key Features

- Byte Pair Encoding (BPE) Tokenization
- Token and Positional Embeddings
- Multi-Head Self-Attention
- Feed-Forward Networks (FFN)
- Layer Normalization
- Residual Connections
- Stacked Transformer Blocks
- Training Loop
- Autoregressive Text Generation

This project emphasizes **correctness, readability, and architectural clarity** over performance optimizations.

---

## System Architecture

The MiniGPT model follows the standard transformer language modeling pipeline:

Raw Text

↓

Tokenizer (BPE)

↓

Token IDs

↓

Embedding + Positional Encoding

↓

Transformer Blocks (N Layers)

↓

Language Model Head (Linear Layer)

↓

Output Tokens


### Architecture Flowchart

+----------------+
| Raw Text |
+--------+-------+
|
v
+--------+-------+
| Tokenizer (BPE)|
+--------+-------+
|
v
+----------------+
| Token IDs |
+--------+-------+
|
v
+-----------------------+
| Embedding + Position |
+-----------+-----------+
|
v
+-----------------------+
| Transformer Block(s) |
| + MHA |
| + FFN |
| + Residuals |
+-----------+-----------+
|
v
+-----------------------+
| LM Head (Linear Layer)|
+-----------+-----------+
|
v
+----------------+
| Output Tokens |
+----------------+


---

## Detailed Components

### Tokenizer (BPE)

**Purpose:**  
Convert raw text into token IDs using Byte Pair Encoding.

**Core Responsibilities:**
- Build vocabulary
- Train merge rules
- Encode strings into token IDs
- Decode token IDs back into text

**Primary Class:**
- `BPETokenizer`
  - `train()`
  - `build_vocab()`
  - `encode()`
  - `decode()`

---

### Embedding Layer

**Purpose:**
- Convert token IDs into dense vector representations
- Inject positional information into token embeddings

**Primary Class:**
- `EmbeddingLayer`
  - Token embeddings
  - Positional embeddings
  - `forward()`

---

### Multi-Head Self-Attention

**Purpose:**  
Enable each token to attend to all previous tokens in the sequence, capturing contextual dependencies.

**Detailed Flow:**

Input X
|
+--+--+
|Linear| → Q
+-----+
|Linear| → K
+-----+
|Linear| → V
+--+--+
|
v
Attention(Q, K, V)
|
v
Concatenate Heads
|
v
Output Linear Layer


**Primary Classes:**
- `AttentionHead`
- `MultiHeadAttention`

---

### Feed-Forward Network

**Purpose:**  
Apply a position-wise two-layer MLP to each token independently.

**Primary Class:**
- `FeedForward`

---

### Layer Normalization and Residual Connections

Each transformer block follows the standard residual pattern:

x = x + MultiHeadAttention(x)
x = x + FeedForward(x)


**Primary Class:**
- `LayerNorm`

---

## Transformer Block

A single transformer block consists of the following structure:

+-------------------+
| Input (X) |
+---------+---------+
|
v
+-------------------+
| LayerNorm |
+-------------------+
|
v
+-------------------+
| Multi-Head Attn |
+-------------------+
|
v
+-------------------+
| Residual Add |
+-------------------+
|
v
+-------------------+
| LayerNorm |
+-------------------+
|
v
+-------------------+
| Feed-Forward |
+-------------------+
|
v
+-------------------+
| Output (X') |
+-------------------+


**Primary Class:**
- `TransformerBlock`
  - `forward()`

---

## Training Procedure

The training process follows these steps:

1. Load dataset
2. Train the tokenizer
3. Convert raw text into token IDs
4. Construct training batches
5. Perform forward pass
6. Compute loss
7. Backpropagation
8. Update model parameters
9. Save trained model

---

## Text Generation

MiniGPT supports autoregressive text generation.

### Generation Algorithm

1. Start with prompt tokens
2. Repeatedly:
   - Feed tokens into the model
   - Obtain logits for the next token
   - Select next token (greedy or sampling)
   - Append token to the sequence

---

## End-to-End Pipeline

Raw Text
↓
Tokenizer (BPE)
↓
Token IDs
↓
Embedding + Position
↓
Transformer Block 1
↓
Transformer Block 2
↓
...
↓
Transformer Block N
↓
Linear Projection
↓
Softmax
↓
Next Token Prediction


---

## Implementation Steps

1. Implement Tokenizer (BPE)
2. Create Dataset and DataLoader
3. Implement mathematical helpers (e.g., softmax, matrix multiplication)
4. Implement Embedding Layer
5. Implement Positional Encoding
6. Implement Attention Head
7. Implement Multi-Head Attention
8. Implement Feed-Forward Network
9. Implement Transformer Block
10. Implement Transformer Model
11. Implement Training Loop
12. Implement Text Generation

---

## Future Improvements

Planned enhancements include:

- Dropout regularization
- Improved weight initialization strategies
- Advanced optimizers (e.g., AdamW)
- GPT-style decoding (top-k, top-p)
- Attention masking
- Training metrics and evaluation tools

---

## Use Cases

- Educational reference for transformer architectures
- Learning GPT-style models from first principles
- Experimentation with custom transformer designs
- Foundation for more advanced language modeling projects

---

## License

This project is released for educational and research purposes.  
Please review the repository for license details before commercial use.

---

**Author:** Suraj Sedai  
**Project:** MiniGPT – A Minimal Transformer Language Model
