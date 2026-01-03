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
from torch.amp import autocast, GradScaler
from datasets import load_dataset

from tokenizer import BPETokenizer

# =====================================================
# CONFIG
# =====================================================
CONFIG = {
    "vocab_size": 2000,
    "max_len": 256,
    "embed_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "ff_hidden_dim": 2048,
    "batch_size": 16,
    "seq_len": 128,
    "epochs": 5,
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
}

CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer.json")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "config.json")

HEARTBEAT_SECONDS = 60
NUM_WORKERS = 0  # REQUIRED on Windows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


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
            num_layers=CONFIG["num_layers"],
        )

        self.ln = nn.LayerNorm(CONFIG["embed_dim"])
        self.head = nn.Linear(CONFIG["embed_dim"], vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.transformer(x)
        x = self.ln(x)
        return self.head(x)


# =====================================================
# HEARTBEAT
# =====================================================
def heartbeat(stop_event, message):
    while not stop_event.is_set():
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
        stop_event.wait(HEARTBEAT_SECONDS)


# =====================================================
# SAVE / LOAD
# =====================================================
def save_checkpoint(model, optimizer, tokenizer, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        MODEL_PATH,
    )

    tokenizer.save(TOKENIZER_PATH)

    with open(CONFIG_PATH, "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"\nCheckpoint saved to `{CHECKPOINT_DIR}/`\n")


def checkpoint_exists():
    return (
        os.path.exists(MODEL_PATH)
        and os.path.exists(TOKENIZER_PATH)
        and os.path.exists(CONFIG_PATH)
    )


# =====================================================
# MAIN
# =====================================================
def main():
    print("Using device:", DEVICE)

    # ---------------------------------------------
    # IF MODEL EXISTS → DO NOT TRAIN AGAIN
    # ---------------------------------------------
    if checkpoint_exists():
        print("\nTrained model already exists.")
        print("Training skipped.")
        print(" You can now use this model for inference.\n")
        return

    # ---------------------------------------------
    # LOAD DATA
    # ---------------------------------------------
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(dataset["train"]["text"])
    print("Training text length:", len(train_text))

    # ---------------------------------------------
    # TOKENIZER
    # ---------------------------------------------
    tokenizer = BPETokenizer(vocab_size=CONFIG["vocab_size"])

    print("Training tokenizer (10–20 minutes)...")
    stop_event = threading.Event()
    hb = threading.Thread(
        target=heartbeat,
        args=(stop_event, "Tokenizer still running..."),
        daemon=True,
    )
    hb.start()

    start = time.time()
    tokenizer.train(train_text)
    stop_event.set()
    hb.join()

    print(f"Tokenizer finished in {(time.time() - start) / 60:.2f} minutes")

    vocab_size = len(tokenizer.vocab)
    print("Actual vocab size:", vocab_size)

    # ---------------------------------------------
    # DATASET
    # ---------------------------------------------
    tokens = tokenizer.encode(train_text)
    dataset = LanguageModelDataset(tokens, CONFIG["seq_len"])

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    print("Batches per epoch:", len(dataloader))

    # ---------------------------------------------
    # MODEL
    # ---------------------------------------------
    model = TransformerLM(vocab_size).to(DEVICE)
    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters()) // 1_000_000,
        "M",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    # ---------------------------------------------
    # TRAINING
    # ---------------------------------------------
    print("\nStarting training...\n")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        last_print = time.time()

        for step, (x, y) in enumerate(dataloader):
            if time.time() - last_print >= HEARTBEAT_SECONDS:
                print(f"[{time.strftime('%H:%M:%S')}] Training still running...")
                last_print = time.time()

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(x)
                loss = criterion(
                    logits.view(-1, vocab_size),
                    y.view(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CONFIG["grad_clip"],
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 500 == 0:
                print(
                    f"Epoch {epoch+1}/{CONFIG['epochs']} | "
                    f"Step {step}/{len(dataloader)} | "
                    f"Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        ppl = math.exp(avg_loss)
        print(
            f"\nEpoch {epoch+1} finished | "
            f"Avg Loss {avg_loss:.4f} | PPL {ppl:.2f}\n"
        )

    # ---------------------------------------------
    # SAVE
    # ---------------------------------------------
    save_checkpoint(model, optimizer, tokenizer, CONFIG["epochs"])
    print("Training complete!")


# =====================================================
# WINDOWS SAFE ENTRY
# =====================================================
if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()
