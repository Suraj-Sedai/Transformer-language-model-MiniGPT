import os, time, math, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.transformer_model import TransformerModel
from tokenizer import BPETokenizer


# -----------------------
# Repro
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, vocab_size, device):
    model.eval()
    loss_sum, tok_sum = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum")
        loss_sum += loss.item()
        tok_sum += y.numel()
    avg = loss_sum / max(1, tok_sum)
    ppl = math.exp(avg) if avg < 20 else float("inf")
    return avg, ppl


# -----------------------
# Fast dataset (no per-item tensor creation)
# -----------------------
class WindowDataset(Dataset):
    def __init__(self, ids_tensor: torch.LongTensor, seq_len: int, start: int, end: int):
        self.ids = ids_tensor
        self.seq_len = seq_len
        self.start = start
        self.end = end
        self.n = max(0, (end - start) - seq_len - 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        idx = self.start + i
        x = self.ids[idx : idx + self.seq_len]
        y = self.ids[idx + 1 : idx + self.seq_len + 1]
        return x, y


# -----------------------
# Corpus helper
# -----------------------
def ensure_reasonable_corpus(text: str, min_tokens_target: int = 300_000) -> str:
    """
    If corpus is tiny, we expand it (for training mechanics + GPU throughput).
    NOTE: This is NOT "real dataset quality" but it prevents toy-scale training.
    """
    # If you want *real* quality, replace this with a real dataset later.
    if len(text) < 10_000:
        # add diversity
        extra = (
            "\n\nThis corpus was automatically expanded for training stability. "
            "It contains varied sentences, topics, and structures.\n"
            "We study language modeling, optimization, and generalization.\n"
            "Transformers use attention, residuals, and normalization.\n"
            "Neural networks learn statistical patterns from data.\n"
            "Software engineering practices improve reliability.\n"
        )
        text = (text + extra) * 50

    # crude expansion if still too small
    # (token count depends on tokenizer; we expand length as proxy)
    while len(text) < min_tokens_target:
        text = text + "\n" + text  # doubles size

    return text


def main():
    set_seed(42)

    # -----------------------
    # GPU / speed switches
    # -----------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 Training on:", device)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # -----------------------
    # Hyperparameters (good defaults for RTX 5070 Ti)
    # -----------------------
    vocab_size_target = 8000

    # We'll auto-adjust seq_len based on corpus size, but start strong:
    seq_len = 256
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    ff_hidden_dim = 1024
    max_len = seq_len

    batch_size = 64              # try 128 if you have VRAM
    grad_accum_steps = 2         # effective batch = 128
    epochs = 10
    lr = 3e-4
    weight_decay = 1e-2
    grad_clip = 1.0
    warmup_steps = 500
    val_ratio = 0.05
    use_amp = True

    num_workers = 4
    print_every = 1

    # -----------------------
    # Paths
    # -----------------------
    corpus_path = "data/corpus.txt"
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    best_path = "checkpoints/best.pt"
    last_path = "checkpoints/last.pt"

    # -----------------------
    # Load corpus
    # -----------------------
    if os.path.exists(corpus_path):
        text = open(corpus_path, "r", encoding="utf-8").read()
    else:
        text = "Artificial intelligence is transforming technology."

    # Expand if tiny (so training becomes meaningful + GPU is used)
    text = ensure_reasonable_corpus(text, min_tokens_target=600_000)

    # -----------------------
    # Tokenizer
    # -----------------------
    tokenizer = BPETokenizer(vocab_size=vocab_size_target)
    tokenizer.train(text)

    vocab_size = len(tokenizer.vocab)
    print("📌 Final vocab size:", vocab_size)

    ids = tokenizer.encode(text)
    total_tokens = len(ids)
    print("📊 Total corpus tokens:", total_tokens)

    # -----------------------
    # Auto-adjust seq_len so we have enough windows
    # -----------------------
    # target: at least 50k train windows
    target_windows = 50_000
    # ensure val has enough tokens too
    while (total_tokens - seq_len - 1) < (target_windows + seq_len + 10) and seq_len > 32:
        seq_len //= 2

    max_len = seq_len
    print(f"✅ Using seq_len={seq_len}")

    # -----------------------
    # Window split
    # -----------------------
    split = int((1 - val_ratio) * total_tokens)
    # ensure val chunk is valid
    split = min(split, total_tokens - (seq_len + 2))
    split = max(split, seq_len + 2)

    ids_t = torch.tensor(ids, dtype=torch.long)

    train_ds = WindowDataset(ids_t, seq_len, 0, split)
    val_ds = WindowDataset(ids_t, seq_len, split, total_tokens)

    if len(val_ds) < 500:
        print(f"⚠️ Val windows are low ({len(val_ds)}). Consider more data for better validation.")

    print("Train windows:", len(train_ds), "| Val windows:", len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=(device == "cuda"), num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        pin_memory=(device == "cuda"), num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )

    # -----------------------
    # Model
    # -----------------------
    model = TransformerModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_hidden_dim=ff_hidden_dim,
        max_len=max_len
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Params: {num_params:,} ≈ {num_params/1e6:.2f}M")

    if device == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            print("⚡ torch.compile enabled")
        except Exception as e:
            print("⚠️ torch.compile failed:", e)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device == "cuda"))

    # -----------------------
    # Scheduler: warmup + cosine
    # -----------------------
    steps_per_epoch = max(1, len(train_loader) // grad_accum_steps)
    total_steps = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -----------------------
    # Resume
    # -----------------------
    start_epoch = 1
    best_val = float("inf")
    global_step = 0

    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        best_val = ckpt.get("best_val", best_val)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"🔁 Resumed from {last_path} (epoch {start_epoch})")

    # -----------------------
    # Train
    # -----------------------
    patience = 2
    bad_epochs = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()

        loss_sum = 0.0
        tok_sum = 0

        optimizer.zero_grad(set_to_none=True)

        for it, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="mean")
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            loss_sum += (loss.item() * grad_accum_steps) * y.numel()
            tok_sum += y.numel()

            if it % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        train_loss = loss_sum / max(1, tok_sum)
        train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")

        val_loss, val_ppl = evaluate(model, val_loader, vocab_size, device)

        dt = time.time() - t0
        tok_s = tok_sum / max(1e-9, dt)
        cur_lr = optimizer.param_groups[0]["lr"]

        if epoch % print_every == 0:
            print(
                f"Epoch {epoch:03d} | lr {cur_lr:.2e} | "
                f"train {train_loss:.4f} ppl {train_ppl:.2f} | "
                f"val {val_loss:.4f} ppl {val_ppl:.2f} | "
                f"{tok_s:,.0f} tok/s"
            )

        # Save last
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "best_val": best_val,
            },
            last_path
        )

        # Save best + early stop
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tokenizer_vocab": tokenizer.vocab,
                    "config": dict(
                        vocab_size=vocab_size,
                        seq_len=seq_len,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        ff_hidden_dim=ff_hidden_dim,
                        max_len=max_len,
                    ),
                    "best_val": best_val,
                },
                best_path
            )
            print(f"✅ Saved best to {best_path} (val loss {best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"🛑 Early stopping (no val improvement for {patience} epochs)")
                break

    # -----------------------
    # Generation
    # -----------------------
    model.eval()
    prompt = "Once upon a time in a small village,"
    ids_prompt = tokenizer.encode(prompt)
    gen_ids = model.generate(ids_prompt, max_new_tokens=80, temperature=0.9, top_k=30, device=device)
    print("\nPROMPT:", prompt)
    print("GENERATED:", tokenizer.decode(gen_ids))


if __name__ == "__main__":
    main()
