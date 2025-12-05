from model.transformer_block import TransformerBlock
from model.embeddings import TokenEmbedding, PositionalEncoding
from utils import matmul, add_vectors, random_matrix, add_vectors_list
import numpy as np

class TransformerModel:
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_hidden_dim, max_len=128):
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(max_len, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, ffn_hidden_dim)
                       for _ in range(num_layers)]
        self.Wo = random_matrix((embed_dim, vocab_size))
        self.bo = np.zeros(vocab_size)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        
    def forward_with_hidden(self, ids):
        """
        Returns:
            logits: [seq_len, vocab_size]
            hidden: [seq_len, embed_dim]  # last hidden vectors before final linear
        """
        # --- Build input embeddings ---
        token_vectors = self.token_embed.forward(ids)  # list of [embed_dim]
        pos_vectors = self.pos_embed.forward(ids)      # list of [embed_dim]

        # Add token + position embeddings
        X = [ [t + p for t,p in zip(tv,pv)] for tv,pv in zip(token_vectors,pos_vectors) ]

        # --- Pass through transformer blocks ---
        for block in self.blocks:
            X = block.forward(X)

        hidden = [v.copy() for v in X]  # last hidden vectors

        # Final linear layer
        logits = []
        for vec in X:
            logits.append(matmul(vec, self.Wo) + self.bo)  # use self.Wo + self.bo

        return logits, hidden

    def forward(self, ids):
        # ids: list of token IDs
        tok_vecs = self.token_embed.forward(ids)
        pos_vecs = self.pos_embed.forward(ids)
        X = [add_vectors(tok_vecs[i], pos_vecs[i]) for i in range(len(ids))]  # seq_len x embed_dim

        # pass through blocks
        for block in self.blocks:
            X = block.forward(X)  # output: seq_len x embed_dim

        # final linear layer
        logits = np.matmul(X, self.Wo) + self.bo  # seq_len x vocab_size

        return logits

    def generate(self, idx, max_new_tokens, tokenizer, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            logits = self.forward(idx)
            last_logits = logits[-1]

            # temperature
            scaled_logits = last_logits / temperature
            exps = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exps / np.sum(exps)

            # top-k
            if top_k is not None:
                probs_idx = probs.argsort()[::-1][:top_k]
                top_probs = probs[probs_idx]
                top_probs /= top_probs.sum()
                next_id = np.random.choice(probs_idx, p=top_probs)
            # top-p
            elif top_p is not None:
                next_id = self.top_p_sample(probs, top_p)
            else:
                next_id = np.random.choice(len(probs), p=probs)

            idx.append(next_id)

        return idx

    @staticmethod
    def top_p_sample(probs, p=0.9):
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumulative_probs = np.cumsum(sorted_probs)

        # keep tokens until cumulative probability < p
        keep_idx = cumulative_probs <= p
        # ensure at least one token
        if not keep_idx.any():
            keep_idx[0] = True

        selected_idx = sorted_idx[keep_idx]
        selected_probs = probs[selected_idx]
        selected_probs /= selected_probs.sum()
        return np.random.choice(selected_idx, p=selected_probs)

        

class MiniTransformer:
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_hidden_dim, num_layers):
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(max_len, embed_dim)
        
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ]

    def forward(self, token_ids):
        tok = self.token_embed.forward(token_ids)
        pos = self.pos_embed.forward(token_ids)
        
        X = add_vectors_list(tok, pos)

        for block in self.blocks:
            X = block.forward(X)

        return X
