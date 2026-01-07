# scripts/train.py
import os
import math
import time
import json
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    "vocab_size": 30000,       # keep your target
    "max_len": 384,
    "embed_dim": 1024,
    "num_heads": 16,
    "num_layers": 12,
    "ff_hidden_dim": 4096,
    "dropout": 0.1,

    "batch_size": 8,
    "seq_len": 256,
    "epochs": 5,
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    "grad_accum_steps": 4,

    # dataloader
    "num_workers": 4,         
}

CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer.json")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "config.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEARTBEAT_SECONDS = 1800


# =====================================================
# TEXT CLEANING
# =====================================================
def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or (line.startswith("=") and line.endswith("=")):
            continue
        # keep case normalization in tokenizer, but it doesn't hurt here either
        lines.append(line.lower())
    return "\n".join(lines)


# =====================================================
# DATASET
# =====================================================
class LanguageModelDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# =====================================================
# MODEL (GPT-style, from scratch)
# =====================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, max_len):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:T, :T]
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, hidden_dim, dropout, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, max_len)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, CONFIG["embed_dim"])
        self.pos_emb = nn.Embedding(CONFIG["max_len"], CONFIG["embed_dim"])
        self.drop = nn.Dropout(CONFIG["dropout"])

        self.blocks = nn.ModuleList([
            Block(
                CONFIG["embed_dim"],
                CONFIG["num_heads"],
                CONFIG["ff_hidden_dim"],
                CONFIG["dropout"],
                CONFIG["max_len"],
            )
            for _ in range(CONFIG["num_layers"])
        ])

        self.ln_f = nn.LayerNorm(CONFIG["embed_dim"])
        self.head = nn.Linear(CONFIG["embed_dim"], vocab_size, bias=False)

        # weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        if T > CONFIG["max_len"]:
            raise ValueError(f"seq len {T} > max_len {CONFIG['max_len']}")

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# =====================================================
# HEARTBEAT
# =====================================================
def heartbeat(stop_event, msg):
    while not stop_event.is_set():
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
        stop_event.wait(HEARTBEAT_SECONDS)


# =====================================================
# SAVE / LOAD
# =====================================================
def save_checkpoint(model, optimizer, scheduler, scaler, tokenizer, epoch, global_step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }, MODEL_PATH)

    # Save tokenizer (your from-scratch format)
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"vocab": tokenizer.vocab, "merges": tokenizer.merges},
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"[{time.strftime('%H:%M:%S')}] Checkpoint saved.", flush=True)


def checkpoint_exists():
    return os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(CONFIG_PATH)


# =====================================================
# MAIN
# =====================================================
def main():
    print("Using device:", DEVICE, flush=True)

    if DEVICE == "cuda":
        # better throughput on modern GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if checkpoint_exists():
        print("✅ Model already exists. Training skipped.", flush=True)
        return

    print("Loading WikiText-2...", flush=True)
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = clean_text("\n".join(raw["train"]["text"]))

    tokenizer = BPETokenizer(vocab_size=CONFIG["vocab_size"])

    print("Training tokenizer...", flush=True)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop, "Tokenizer running..."), daemon=True).start()

    t0 = time.time()
    tokenizer.train(train_text)
    stop.set()
    print(
        f"Tokenizer finished in {(time.time()-t0)/60:.1f} min | "
        f"vocab={len(tokenizer.vocab)} | merges={len(tokenizer.merges)}",
        flush=True
    )

    tokens = tokenizer.encode(train_text)
    vocab_size = len(tokenizer.vocab)
    print(f"Encoded tokens: {len(tokens):,} | vocab_size: {vocab_size}", flush=True)

    ds = LanguageModelDataset(tokens, CONFIG["seq_len"])
    if len(ds) == 0:
        raise RuntimeError("Dataset too small. Lower seq_len or check tokenization.")

    loader = DataLoader(
        ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
        persistent_workers=(CONFIG["num_workers"] > 0),
    )

    model = GPTLM(vocab_size).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.95)
    )

    total_steps = (len(loader) // CONFIG["grad_accum_steps"]) * CONFIG["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    criterion = nn.CrossEntropyLoss()

    # AMP: fp16 on cuda
    use_amp = (DEVICE == "cuda")
    scaler = GradScaler(enabled=use_amp)

    print("\nStarting training...\n", flush=True)
    global_step = 0
    tokens_per_step = CONFIG["batch_size"] * CONFIG["seq_len"] * CONFIG["grad_accum_steps"]
    start_time = time.time()
    last_log = time.time()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=use_amp):
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
                global_step += 1

            epoch_loss += loss.item() * CONFIG["grad_accum_steps"]

            # log every 60 seconds
            if time.time() - last_log >= 60:
                elapsed = time.time() - start_time
                tps = (global_step * tokens_per_step) / max(1e-9, elapsed)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"epoch {epoch+1}/{CONFIG['epochs']} | step {global_step}/{total_steps} | "
                    f"loss {loss.item()*CONFIG['grad_accum_steps']:.4f} | lr {lr:.2e} | "
                    f"tokens/s {tps:,.0f}",
                    flush=True
                )
                last_log = time.time()

        avg_loss = epoch_loss / len(loader)
        ppl = math.exp(min(20, avg_loss))
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} | avg loss {avg_loss:.4f} | ppl {ppl:.2f}\n", flush=True)

    save_checkpoint(model, optimizer, scheduler, scaler, tokenizer, CONFIG["epochs"], global_step)
    print("Training complete!", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
