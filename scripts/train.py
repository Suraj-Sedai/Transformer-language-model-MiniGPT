import math
import time
import threading
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from tokenizer import BPETokenizer


# -------------------------
# CONFIG
# -------------------------
vocab_size = 2000
max_len = 256
embed_dim = 512
num_heads = 8
num_layers = 6
ff_hidden_dim = 2048
batch_size = 16
seq_len = 128
epochs = 5

lr = 3e-4
weight_decay = 1e-2
grad_clip = 1.0

num_workers = 0  # REQUIRED on Windows

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# DATASET
# -------------------------
class LanguageModelDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# -------------------------
# MODEL
# -------------------------
class TransformerLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.transformer(x)
        x = self.ln(x)
        return self.head(x)


# -------------------------
# HEARTBEAT THREAD
# -------------------------
def heartbeat(stop_event, message):
    while not stop_event.is_set():
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
        stop_event.wait(60)


# -------------------------
# MAIN
# -------------------------
def main():
    print("Using device:", device)

    # -------------------------
    # LOAD DATA
    # -------------------------
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(dataset["train"]["text"])
    print("Training text length:", len(train_text))

    # -------------------------
    # TOKENIZER (WITH HEARTBEAT)
    # -------------------------
    tokenizer = BPETokenizer(vocab_size=vocab_size)

    print("Training tokenizer (this WILL take 10–20 minutes)...")

    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=heartbeat,
        args=(stop_event, "Tokenizer still running..."),
        daemon=True,
    )
    hb_thread.start()

    start_time = time.time()
    tokenizer.train(train_text)
    stop_event.set()
    hb_thread.join()

    print(f"Tokenizer finished in {(time.time() - start_time) / 60:.2f} minutes")

    actual_vocab_size = len(tokenizer.vocab)
    print("Actual vocab size:", actual_vocab_size)

    # -------------------------
    # ENCODE
    # -------------------------
    token_ids = tokenizer.encode(train_text)
    print("Total tokens:", len(token_ids))

    # -------------------------
    # DATASET
    # -------------------------
    train_dataset = LanguageModelDataset(token_ids, seq_len)
    print("Training sequences:", len(train_dataset))

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    print("Batches per epoch:", len(dataloader))

    # -------------------------
    # MODEL
    # -------------------------
    model = TransformerLM(actual_vocab_size).to(device)
    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters()) // 1_000_000,
        "M"
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    print("Starting training...")

    # -------------------------
    # TRAINING LOOP
    # -------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        last_print = time.time()

        for step, (x, y) in enumerate(dataloader):
            if time.time() - last_print >= 60:
                print(f"[{time.strftime('%H:%M:%S')}] Training still running...")
                last_print = time.time()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(x)
                loss = criterion(
                    logits.view(-1, actual_vocab_size),
                    y.view(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 500 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Step {step}/{len(dataloader)} | "
                    f"Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        ppl = math.exp(avg_loss)
        print(f"\nEpoch {epoch+1} finished | Avg Loss {avg_loss:.4f} | PPL {ppl:.2f}\n")

    print("Training complete.")


# -------------------------
# WINDOWS SAFE ENTRY
# -------------------------
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
