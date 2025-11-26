from utils import matmul, random_matrix

class TokenEmbedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = random_matrix((vocab_size, embed_dim))
    
    def forward(self, token_ids):
        return [self.W[id] for id in token_ids]

class PosEmbedding:
    def __init__(self, max_len, embed_dim):
        self.P = random_matrix((max_len, embed_dim))
    
    def forward(self, token_ids):
        return [self.P[pos] for pos in range(len(token_ids))]
