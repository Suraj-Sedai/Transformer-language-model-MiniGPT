from utils import matmul, random_matrix

class TokenEmbedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = random_matrix((vocab_size, embed_dim))
    
    def forward(self, token_ids):
        return [self.W[id] for id in token_ids]

# class PosEmbedding:
#     def __init__(self, max_len, embed_dim):
#         self.P = random_matrix((max_len, embed_dim))
#
#     def forward(self, token_ids):
#         return [self.P[pos] for pos in range(len(token_ids))]

import numpy as np

class PositionalEncoding:
    """
    Sinusoidal positional encoding.
    Precompute a (max_len x d_model) matrix and return slices.
    """
    def __init__(self, max_len: int, d_model: int):
        self.max_len = max_len
        self.d_model = d_model
        self.P = self._build_positional_encoding(max_len, d_model)  # shape (max_len, d_model)

    def _build_positional_encoding(self, max_len, d_model):
        # Create matrix of shape (max_len, d_model)
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        # position indices (0..max_len-1)
        pos = np.arange(max_len)[:, np.newaxis]                      # shape (max_len, 1)
        # dimension indices (0..d_model-1)
        i = np.arange(d_model)[np.newaxis, :]                        # shape (1, d_model)

        # Compute the angle rates: 1 / (10000^(2i/d_model))
        angle_rates = 1.0 / (10000 ** ((2 * (i // 2)) / np.float32(d_model)))
        angles = pos * angle_rates                                   # shape (max_len, d_model)

        # apply sin to even indices, cos to odd indices
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        return pe

    def forward(self, token_ids):
        """
        token_ids: list of token ids (we only need len(token_ids) to get positions)
        returns: list of numpy arrays (seq_len x d_model) OR a numpy array (seq_len, d_model)
        """
        seq_len = len(token_ids)
        # slice precomputed P
        return self.P[:seq_len]   # returns numpy array shape (seq_len, d_model)

if __name__ == "__main__":
    from tokenizer import BPETokenizer
    tok = BPETokenizer(vocab_size=100)
    tok.train("hello world this is a small test corpus for positional encoding")
    ids = tok.encode("hello world this is")

    d_model = 8
    pos_enc = PositionalEncoding(max_len=50, d_model=d_model)
    P = pos_enc.forward(ids)    # shape (seq_len, d_model)
    print("P shape:", P.shape)
    print("First row:", P[0])
    print("Second row:", P[1])
