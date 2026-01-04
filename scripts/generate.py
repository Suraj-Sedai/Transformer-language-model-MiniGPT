# scripts/generate.py
import os
import json
import torch
import torch.nn.functional as F

from scripts.train import (
    TransformerLM,
    CONFIG,
    MODEL_PATH,
    TOKENIZER_PATH,
    CONFIG_PATH,
)

from tokenizer import BPETokenizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = BPETokenizer(vocab_size=len(data["vocab"]))
    tokenizer.vocab = data["vocab"]
    tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.merges = data.get("merges", [])

    return tokenizer


def safe_sample_logits(logits, temperature=1.0, top_k=40):
    logits = logits.float()

    # Temperature
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # Remove NaNs / Infs
    logits = torch.nan_to_num(
        logits, nan=-1e9, posinf=-1e9, neginf=-1e9
    )

    vocab_size = logits.size(-1)

    # Safe top-k
    if top_k is not None and top_k > 0:
        top_k = min(top_k, vocab_size)
        values, indices = torch.topk(logits, top_k, dim=-1)

        filtered = torch.full_like(logits, -1e9)
        filtered.scatter_(1, indices, values)
        logits = filtered

    probs = F.softmax(logits, dim=-1)

    # If probabilities are invalid → greedy fallback
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.9,
    top_k=40,
):
    model.eval()

    token_ids = tokenizer.encode(prompt)

    if len(token_ids) == 0:
        raise ValueError("Prompt produced no tokens")

    ids = torch.tensor(
        token_ids, dtype=torch.long, device=DEVICE
    ).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(ids)
        next_logits = logits[:, -1, :]

        next_token = safe_sample_logits(
            next_logits,
            temperature=temperature,
            top_k=top_k,
        )

        ids = torch.cat([ids, next_token], dim=1)

        if ids.size(1) >= CONFIG["max_len"]:
            break

    return tokenizer.decode(ids[0].tolist())


def main():
    if not (
        os.path.exists(MODEL_PATH)
        and os.path.exists(TOKENIZER_PATH)
        and os.path.exists(CONFIG_PATH)
    ):
        raise FileNotFoundError("Model files not found. Train first.")

    print("Using device:", DEVICE)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = len(tokenizer.vocab)

    model = TransformerLM(vocab_size).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    print("Model loaded successfully\n")

    while True:
        prompt = input("Prompt (empty to quit): ").strip()
        if not prompt:
            break

        try:
            output = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=150,
            )
            print("\n--- Generated ---")
            print(output)
            print("-----------------\n")

        except Exception as e:
            print("Generation error:", e)


if __name__ == "__main__":
    main()
