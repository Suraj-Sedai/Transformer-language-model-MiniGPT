# model/embeddings.py
import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Simple token embedding (nn.Embedding)."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # optional: initialize
        nn.init.normal_(self.embed.weight, mean=0.0, std=embed_dim ** -0.5)

    def forward(self, token_ids):
        # token_ids: (B, T) or (T,) where B=1
        return self.embed(token_ids)  # -> (B, T, D)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (deterministic)."""
    def __init__(self, max_len, embed_dim):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        if embed_dim % 2 == 1:
            # if odd embed_dim, last column handled safely
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, D)
        self.register_buffer("pe", pe)

    def forward(self, token_ids):
        # token_ids (B, T) -> returns pos enc (B, T, D) sliced to length T
        if token_ids.ndim == 1:
            T = token_ids.shape[0]
            return self.pe[:, :T, :].clone()
        else:
            B, T = token_ids.shape
            return self.pe[:, :T, :].expand(B, -1, -1).clone()
