# scripts/train.py
import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer_model import TransformerModel
from tokenizer import BPETokenizer  # your existing tokenizer file
import numpy as np

# -----------------------
# Config / Hyperparams
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 1000        # after tokenizer.train -> tokenizer.vocab_size (or use this as arg)
max_len = 64
embed_dim = 128
num_heads = 4
num_layers = 2
ff_hidden_dim = 512
batch_size = 8
seq_len = 32            # context window for training (should be <= max_len)
epochs = 50
lr = 3e-4
weight_decay = 1e-2
grad_clip = 1.0
print_every = 10

# -----------------------
# Prepare data & tokenizer
# -----------------------
text = ("Artificial Intelligence (AI) is transforming technology. "
        "Machine Learning (ML) is a key part of AI. "
        "Neural networks are a type of ML model. "
        "AI can perform tasks like language understanding, image recognition, "
        "and data analysis. "
        "Generative AI can create text, images, and music automatically. "
        "Reinforcement Learning trains agents to take actions to maximize rewards. "
        "Deep learning models learn hierarchical representations of data. "
        "Natural Language Processing (NLP) enables machines to understand human language. ") * 200

tokenizer = BPETokenizer(vocab_size=vocab_size)
tokenizer.train(text)
vocab_size = len(tokenizer.vocab)  # update actual vocab


ids = tokenizer.encode(text)  # list of ints

# create dataset of sliding windows
X = []
Y = []
for i in range(len(ids) - seq_len):
    X.append(ids[i:i+seq_len])
    Y.append(ids[i+seq_len])
X = np.array(X, dtype=np.int64)
Y = np.array(Y, dtype=np.int64)

# simple batching helper
def get_batch(batch_idx):
    start = batch_idx * batch_size
    end = start + batch_size
    x = torch.tensor(X[start:end], dtype=torch.long).to(device)
    y = torch.tensor(Y[start:end], dtype=torch.long).to(device)
    return x, y

num_batches = len(X) // batch_size

# -----------------------
# Model, loss, optimizer
# -----------------------
model = TransformerModel(vocab_size=vocab_size, max_len=max_len, embed_dim=embed_dim,
                         num_heads=num_heads, num_layers=num_layers, ff_hidden_dim=ff_hidden_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# -----------------------
# Training loop (simple)
# -----------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    perm = np.random.permutation(len(X) // batch_size)
    for bidx in range(num_batches):
        i = bidx  # you can randomize
        xb, yb = get_batch(i)
        optimizer.zero_grad()
        logits = model(xb)  # (B, T, V)
        # we only predict the last token, consistent with earlier scripts:
        last_logits = logits[:, -1, :]  # (B, V)
        loss = criterion(last_logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    if epoch % print_every == 0:
        avg_loss = total_loss / num_batches
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        print(f"Epoch {epoch} | avg loss {avg_loss:.4f} | ppl {ppl:.2f}")

# -----------------------
# Generate
# -----------------------
model.eval()
prompt = "Once upon a time in a small village,"
ids_prompt = tokenizer.encode(prompt)
output_ids = model.generate(ids_prompt, max_new_tokens=40, temperature=0.9, top_k=30, device=device)
print("PROMPT:", prompt)
print("GENERATED:", tokenizer.decode(output_ids))