# model/transformer_block.py
import torch
import torch.nn as nn
from .layers import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    """
    One transformer block:
      - MultiheadAttention (self-attention)
      - Residual + LayerNorm
      - FeedForward (position-wise)
      - Residual + LayerNorm
    """
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim, dropout=dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, T, D)
        # attn_mask: optional (T, T) or (B, T, T)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)  # (B, T, D)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.ln2(x)
        return x
