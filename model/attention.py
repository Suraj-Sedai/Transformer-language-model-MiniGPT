import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# --------------------------------------
# Self-Attention (used mostly for testing)
# --------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        """
        X: (B, T, D)
        """
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.embed_dim)
        weights = F.softmax(scores, dim=-1)

        out = torch.matmul(weights, V)
        return out, weights


# --------------------------------------
# One Attention Head
# --------------------------------------
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.scale = sqrt(head_dim)

        self.W_q = nn.Linear(embed_dim, head_dim)
        self.W_k = nn.Linear(embed_dim, head_dim)
        self.W_v = nn.Linear(embed_dim, head_dim)

    def forward(self, X):
        """
        X: (B, T, D)
        """
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)

        out = torch.matmul(weights, V)
        return out, weights


# --------------------------------------
# Multi-Head Attention
# --------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, self.head_dim) for _ in range(num_heads)]
        )

        self.Wo = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        """
        X: (B, T, D)
        """
        head_outputs = []
        all_weights = []

        for head in self.heads:
            out, w = head(X)
            head_outputs.append(out)
            all_weights.append(w)

        # concat heads on last dimension
        concat = torch.cat(head_outputs, dim=-1)  # (B, T, embed_dim)

        # final output projection
        out = self.Wo(concat)

        return out, all_weights
