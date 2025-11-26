from model.transformer_block import TransformerBlock
from model.embeddings import TokenEmbedding, PosEmbedding
from utils import matmul, add_vectors, random_matrix, add_vectors_list
import numpy as np

class TransformerModel:
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_hidden_dim, max_len=128):
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embed = PosEmbedding(max_len, embed_dim)
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

    def generate(self, idx, max_new_tokens, tokenizer):
        """
        idx: list of token ids (context)
        max_new_tokens: how many tokens to generate
        tokenizer: your BPE or simple tokenizer
        """

        for _ in range(max_new_tokens):
            # forward pass: logits shape (T, vocab)
            logits = self.forward(idx)

            # take last position
            last_logits = logits[-1]   # shape: (vocab,)

            # convert to probabilities using softmax
            exps = np.exp(last_logits - np.max(last_logits))
            probs = exps / np.sum(exps)

            # sample from distribution (probabilistic)
            next_id = int(np.random.choice(len(probs), p=probs))


            # append prediction
            idx.append(next_id)

        return idx
    

class MiniTransformer:
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_hidden_dim, num_layers):
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embed = PosEmbedding(max_len, embed_dim)
        
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
