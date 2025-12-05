# model/transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import TokenEmbedding, PositionalEncoding
from .transformer_block import TransformerBlock


class TransformerModel(nn.Module):
    """
    Minimal GPT-like model (causal LM)
    - token embedding
    - positional encoding (sinusoidal)
    - stack of transformer blocks
    - linear head to vocab
    """
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, num_layers, ff_hidden_dim, dropout=0.1, tie_weights=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(max_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)  # will tie optionally

        if tie_weights:
            # tie embedding and head weights (common in language models)
            self.head.weight = self.token_embed.embed.weight

        self.max_len = max_len

    def forward(self, token_ids, attn_mask=None):
        """
        token_ids: (B, T) long tensor
        attn_mask: (T, T) boolean mask for causal/other masking (optional)
        returns logits: (B, T, vocab_size)
        """
        device = token_ids.device
        B, T = token_ids.shape

        # embeddings
        tok = self.token_embed(token_ids)           # (B, T, D)
        pos = self.pos_embed(token_ids.to(device))  # (B, T, D)
        x = tok + pos

        # transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.ln_f(x)  # final norm
        logits = self.head(x)  # (B, T, V)
        return logits

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=20, temperature=1.0, top_k=None, top_p=None, device=None):
        """
        Simple autoregressive generation (batch size 1). Supports top-k, top-p and temperature.
        token_ids: list or tensor of shape (T,) or (1, T)
        returns list of token ids (including prompt).
        """
        if device is None:
            device = next(self.parameters()).device

        if isinstance(token_ids, (list, tuple)):
            idx = torch.tensor([token_ids], dtype=torch.long, device=device)
        elif token_ids.ndim == 1:
            idx = token_ids.unsqueeze(0).to(device)
        else:
            idx = token_ids.to(device)

        for _ in range(max_new_tokens):
            T = idx.shape[1]
            if T > self.max_len:
                # keep last max_len tokens
                idx = idx[:, -self.max_len:]

            logits = self.forward(idx)  # (1, T, V)
            last_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            probs = F.softmax(last_logits, dim=-1).view(-1)  # ensure 1D

            # top-k
            if top_k is not None:
                top_k = min(top_k, probs.numel())  # safety check
                topk = torch.topk(probs, top_k)
                choices = topk.indices
                p = topk.values / topk.values.sum()
                next_id = choices[torch.multinomial(p, num_samples=1)].item()

            # top-p (nucleus)
            elif top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                cutoff = (cumulative > top_p).nonzero(as_tuple=False)
                if cutoff.numel() > 0:
                    cutoff_idx = cutoff[0].item()
                    keep = sorted_idx[: cutoff_idx + 1]
                else:
                    keep = sorted_idx
                subp = probs[keep]
                subp = subp / subp.sum()
                next_id = keep[torch.multinomial(subp, 1)].item()
            else:
                next_id = torch.multinomial(probs, num_samples=1).item()

            idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)

        return idx.squeeze(0).tolist()
