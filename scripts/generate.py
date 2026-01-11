# scripts/generate.py
import os
import json
import math
import torch
import torch.nn.functional as F

from scripts.train import (
    GPTLM,          # <-- correct class name from train.py
    CONFIG,
    MODEL_PATH,
    TOKENIZER_PATH,
    CONFIG_PATH,
)

from tokenizer import BPETokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(path: str) -> BPETokenizer:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = BPETokenizer(vocab_size=len(data["vocab"]))
    tokenizer.vocab = data["vocab"]
    tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.merges = data.get("merges", [])
    return tokenizer


def safe_sample_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = 40) -> torch.Tensor:
    """
    logits: (B, V)
    returns next_token: (B, 1)
    """
    logits = logits.float()

    # Greedy if temperature <= 0
    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)

    # Remove NaNs / Infs safely
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)

    vocab_size = logits.size(-1)

    # Top-k filtering
    if top_k is not None and top_k > 0:
        k = min(int(top_k), vocab_size)
        values, indices = torch.topk(logits, k, dim=-1)

        filtered = torch.full_like(logits, -1e9)
        filtered.scatter_(1, indices, values)
        logits = filtered

    probs = F.softmax(logits, dim=-1)

    # Fallback if probs invalid
    if (not torch.isfinite(probs).all()) or (probs.sum(dim=-1).min().item() <= 0):
        return torch.argmax(logits, dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.9,
    top_k: int | None = 40,
) -> str:
    model.eval()

    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        raise ValueError("Prompt produced no tokens (tokenizer.encode returned empty list).")

    ids = torch.tensor(token_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        # Crop context to model's max_len (so we can generate longer sequences safely)
        idx_cond = ids[:, -CONFIG["max_len"] :]

        logits = model(idx_cond)            # (1, Tc, V)
        next_logits = logits[:, -1, :]      # (1, V)

        next_token = safe_sample_logits(
            next_logits,
            temperature=temperature,
            top_k=top_k,
        )                                   # (1, 1)

        ids = torch.cat([ids, next_token], dim=1)

    return tokenizer.decode(ids[0].tolist())


def main():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(CONFIG_PATH)):
        raise FileNotFoundError("Model files not found. Train first (scripts/train.py).")

    print("Using device:", DEVICE)

    # Load config from file (optional, but helps if CONFIG changed in code)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        saved_config = json.load(f)

    # If you want to force using saved config for generation, uncomment:
    # CONFIG.update(saved_config)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = len(tokenizer.vocab)

    model = GPTLM(vocab_size).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Your train.py saves with key "model"
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)

    print("Model loaded successfully ✅\n")

    print("Generation settings:")
    print(f"  max_new_tokens = 150")
    print(f"  temperature    = 0.9")
    print(f"  top_k          = 40")
    print("")

    while True:
        prompt = input("Prompt (empty to quit): ").strip()
        if not prompt:
            break

        try:
            out = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.9,
                top_k=40,
            )
            print("\n--- Generated ---")
            print(out)
            print("-----------------\n")

        except Exception as e:
            print("Generation error:", repr(e))


if __name__ == "__main__":
    main()
 