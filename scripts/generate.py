# scripts/generate.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import BPETokenizer

# =====================================================
# PATHS
# =====================================================
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer.json")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "config.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# MODEL (must match train.py EXACTLY)
# =====================================================
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, config["embed_dim"])
        self.pos_emb = nn.Embedding(config["max_len"], config["embed_dim"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["embed_dim"],
            nhead=config["num_heads"],
            dim_feedforward=config["ff_hidden_dim"],
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["num_layers"],
        )

        self.ln = nn.LayerNorm(config["embed_dim"])
        self.head = nn.Linear(config["embed_dim"], vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.token_emb(x) + self.pos_emb(pos)

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()

        x = self.transformer(x, mask=causal_mask)
        x = self.ln(x)
        return self.head(x)


# =====================================================
# SAMPLING
# =====================================================
def sample_logits(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_k is not None:
        values, _ = torch.topk(probs, top_k)
        min_prob = values[:, -1].unsqueeze(-1)
        probs = torch.where(probs < min_prob, torch.zeros_like(probs), probs)

    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False

        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)

    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


# =====================================================
# GENERATION
# =====================================================
@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
):
    model.eval()

    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -model.pos_emb.num_embeddings :]

        logits = model(tokens_cond)
        next_logits = logits[:, -1, :]

        next_token = sample_logits(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


# =====================================================
# MAIN
# =====================================================
def main():
    if not (
        os.path.exists(MODEL_PATH)
        and os.path.exists(TOKENIZER_PATH)
        and os.path.exists(CONFIG_PATH)
    ):
        raise FileNotFoundError(
            "Model, tokenizer, or config not found. Train first."
        )

    print("Using device:", DEVICE)

    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = BPETokenizer(vocab_size=config["vocab_size"])
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = data.get("merges", None)

    vocab_size = len(tokenizer.vocab)

    # Load model
    model = TransformerLM(vocab_size, config).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    print("Model loaded successfully")

    # Interactive loop
    while True:
        prompt = input("\nPrompt (empty to quit): ").strip()
        if not prompt:
            break

        output = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=150,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )

        print("\n--- Generated Text ---")
        print(output)
        print("----------------------")


if __name__ == "__main__":
    main()
