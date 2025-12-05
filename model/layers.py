# model/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Feed-forward network used after attention (position-wise)."""
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu  # you can change to GELU if you like

    def forward(self, x):
        # x: (B, T, D)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LayerNorm(nn.Module):
    """Wrapper around PyTorch's LayerNorm for consistency."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)
