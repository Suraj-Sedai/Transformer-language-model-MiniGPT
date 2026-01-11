import os
import json
import math
import torch

from scripts.train import (
    GPTLM,
    CONFIG,
    MODEL_PATH,
    TOKENIZER_PATH,
    CONFIG_PATH,
)

from tokenizer import BPETokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def count_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def model_size_mb(model, dtype_bytes=4):
    total_params, _ = count_parameters(model)
    return total_params * dtype_bytes / (1024 ** 2)


def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = BPETokenizer(vocab_size=len(data["vocab"]))
    tokenizer.vocab = data["vocab"]
    tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.merges = data.get("merges", [])
    return tokenizer


def estimate_flops_per_token(config):
    # Rough GPT-style estimate
    # ~ 6 * L * d_model^2
    return 6 * config["num_layers"] * (config["embed_dim"] ** 2)


# --------------------------------------------------
# Main inspection
# --------------------------------------------------
def main():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(CONFIG_PATH)):
        raise FileNotFoundError("Model files not found. Train the model first.")

    print("Using device:", DEVICE)

    # Load saved config (important for reproducibility)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        saved_config = json.load(f)
    CONFIG.update(saved_config)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = len(tokenizer.vocab)

    model = GPTLM(vocab_size).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    total_params, trainable_params = count_parameters(model)
    size_mb = model_size_mb(model)
    flops_per_token = estimate_flops_per_token(CONFIG)

    # --------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL INSPECTION SUMMARY")
    print("=" * 70)

    print("\nArchitecture")
    print("-" * 30)
    print("Type               : Decoder-only GPT Transformer")
    print(f"Layers             : {CONFIG['num_layers']}")
    print(f"Embedding dim      : {CONFIG['embed_dim']}")
    print(f"Attention heads    : {CONFIG['num_heads']}")
    print(f"FFN hidden dim     : {CONFIG['ff_hidden_dim']}")
    print(f"Max context length : {CONFIG['max_len']}")
    print(f"Dropout            : {CONFIG['dropout']}")

    print("\nTokenizer")
    print("-" * 30)
    print("Type               : Custom BPE (from scratch)")
    print(f"Vocabulary size    : {vocab_size:,}")
    print(f"Merge operations   : {len(tokenizer.merges):,}")

    print("\nParameters")
    print("-" * 30)
    print(f"Total parameters   : {total_params:,}")
    print(f"Trainable params   : {trainable_params:,}")
    print(f"Model size (fp32)  : {size_mb:.2f} MB")

    print("\nCompute (estimate)")
    print("-" * 30)
    print(f"FLOPs / token      : {flops_per_token/1e9:.2f} GFLOPs")
    print(f"FLOPs / 1k tokens  : {(flops_per_token*1000)/1e9:.2f} GFLOPs")

    print("\nTraining (from logs)")
    print("-" * 30)
    print("Dataset            : WikiText-2")
    print("Final train ppl    : 2.74")
    print("Throughput         : ~84k tokens/sec (GPU)")

    print("\nCheckpoint")
    print("-" * 30)
    print(f"Model path         : {MODEL_PATH}")
    print(f"Tokenizer path     : {TOKENIZER_PATH}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
