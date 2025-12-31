import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.transformer_model import TransformerModel
from tokenizer import BPETokenizer


# -----------------------
# Hyperparameters
# -----------------------
vocab_size = 5000
max_len = 64
embed_dim = 128
num_heads = 4
num_layers = 2
ff_hidden_dim = 512
batch_size = 16          # better for GPU
seq_len = 64             # full window
epochs = 70

lr = 3e-4
weight_decay = 1e-2
grad_clip = 1.0
print_every = 1


# -----------------------
# Dataset
# -----------------------
class TextDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y


# -----------------------
# Load / Build Corpus
# -----------------------
corpus_path = "data/corpus.txt"

if os.path.exists(corpus_path):
    text = open(corpus_path, "r", encoding="utf-8").read()
else:
    print("⚠️  No corpus found — using default repeated AI text.")
    base_text = (
        "Artificial intelligence is transforming technology. "
        "Machine learning is a key part of AI. "
        "Neural networks learn patterns from data. "
        "AI models generate text and understand language. "
    )
    text = base_text * 300  # repeated to build a large learning set


# -----------------------
# Tokenizer
# -----------------------
tokenizer = BPETokenizer(vocab_size=vocab_size)
tokenizer.train(text)

vocab_size = len(tokenizer.vocab)
print("📌 Final vocab size:", vocab_size)

ids = tokenizer.encode(text)
dataset = TextDataset(ids, seq_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# -----------------------
# Model + Optimizer
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Training on:", device)

model = TransformerModel(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ff_hidden_dim=ff_hidden_dim,
    max_len=max_len
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# -----------------------
# Training Loop
# -----------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)                     # (batch, seq_len, vocab)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    if epoch % print_every == 0:
        avg_loss = total_loss / len(loader)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | ppl {ppl:.2f}")


# -----------------------
# Generation Test
# -----------------------
model.eval()

prompt = "Once upon a time in a small village,"
ids_prompt = tokenizer.encode(prompt)

print("\nPROMPT:", prompt)

generated_ids = model.generate(
    ids_prompt,
    max_new_tokens=80,
    temperature=0.9,
    top_k=30,
    device=device
)

generated_text = tokenizer.decode(generated_ids)
print("GENERATED:", generated_text)

num_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {num_params:,} (trainable: {trainable:,}) ≈ {num_params/1e6:.2f}M")

