# scripts/train.py
import os
import math
import time
import json
import threading

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from datasets import load_dataset

from tokenizer import BPETokenizer

# =====================================================
# CONFIG
# =====================================================
CONFIG = {
    "vocab_size": 30000,      # larger vocab for better text
    "max_len": 384,            # longer context window
    "embed_dim": 1024,         # larger embedding
    "num_heads": 16,           # multi-head attention
    "num_layers": 12,          # deeper transformer
    "ff_hidden_dim": 4096,     # larger feed-forward
    "batch_size": 8,           # batch per GPU step
    "seq_len": 256,
    "epochs": 5,
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    "grad_accum_steps": 4,     # effective larger batch
}

CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer.json")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "config.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
HEARTBEAT_SECONDS = 60

# =====================================================
# TEXT CLEANING
# =====================================================
def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or (line.startswith("=") and line.endswith("=")):
            continue
        lines.append(line.lower())
    return " ".join(lines)

# =====================================================
# DATASET
# =====================================================
class LanguageModelDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)

# =====================================================
# MODEL
# =====================================================
class TransformerLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, CONFIG["embed_dim"])
        self.pos_emb = nn.Embedding(CONFIG["max_len"], CONFIG["embed_dim"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG["embed_dim"],
            nhead=CONFIG["num_heads"],
            dim_feedforward=CONFIG["ff_hidden_dim"],
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=CONFIG["num_layers"]
        )
        self.ln = nn.LayerNorm(CONFIG["embed_dim"])
        self.head = nn.Linear(CONFIG["embed_dim"], vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        # causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(CONFIG["max_len"], CONFIG["max_len"]), diagonal=1).bool()
        )

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        mask = self.causal_mask[:T, :T]
        x = self.transformer(x, mask=mask)
        x = self.ln(x)
        return self.head(x)

# =====================================================
# HEARTBEAT
# =====================================================
def heartbeat(stop_event, msg):
    while not stop_event.is_set():
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")
        stop_event.wait(HEARTBEAT_SECONDS)

# =====================================================
# SAVE / LOAD
# =====================================================
def save_checkpoint(model, optimizer, scheduler, tokenizer, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }, MODEL_PATH)

    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        json.dump({"vocab": tokenizer.vocab, "merges": getattr(tokenizer, "merges", None)}, f, indent=2, ensure_ascii=False)

    with open(CONFIG_PATH, "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"[{time.strftime('%H:%M:%S')}] Checkpoint saved.")

def checkpoint_exists():
    return os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(CONFIG_PATH)

# =====================================================
# MAIN
# =====================================================
def main():
    print("Using device:", DEVICE)

    if checkpoint_exists():
        print("✅ Model already exists. Training skipped.")
        return

    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = clean_text("\n".join(dataset["train"]["text"]))

    tokenizer = BPETokenizer(vocab_size=CONFIG["vocab_size"])

    print("Training tokenizer (this may take 20+ minutes)...")
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop, "Tokenizer running..."), daemon=True).start()
    tokenizer.train(train_text)
    stop.set()

    tokens = tokenizer.encode(train_text)
    vocab_size = len(tokenizer.vocab)
    dataset = LanguageModelDataset(tokens, CONFIG["seq_len"])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"), drop_last=True)

    model = TransformerLM(vocab_size).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    total_steps = (len(loader) // CONFIG["grad_accum_steps"]) * CONFIG["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(DEVICE=="cuda"))

    print("Starting training...\n")
    step_count = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        last_print = time.time()

        for step, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with autocast(enabled=(DEVICE=="cuda")):
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss = loss / CONFIG["grad_accum_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % CONFIG["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step_count += 1

            total_loss += loss.item() * CONFIG["grad_accum_steps"]

            # Heartbeat print
            if time.time() - last_print >= HEARTBEAT_SECONDS:
                print(f"[{time.strftime('%H:%M:%S')}] Training in progress...")
                last_print = time.time()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss {avg_loss:.4f} | PPL {math.exp(avg_loss):.2f}")

    save_checkpoint(model, optimizer, scheduler, tokenizer, CONFIG["epochs"])
    print("Training complete!")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
