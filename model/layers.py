import numpy as np

class FeedForward:
    def __init__(self, embed_dim, hidden_dim):
        self.W1 = np.random.uniform(-0.1, 0.1, (embed_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.uniform(-0.1, 0.1, (hidden_dim, embed_dim))
        self.b2 = np.zeros(embed_dim)


    def forward(self, X):
        # X shape: (B, T, D)
        hidden = np.matmul(X, self.W1) + self.b1
        hidden = np.maximum(hidden, 0)   # ReLU
        out = np.matmul(hidden, self.W2) + self.b2
        return out


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps
        # Learnable parameters (gamma, beta)
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim

    def forward(self, X):
        # X is list of vectors: [seq_len][dim]
        out = []
        for vec in X:
            mean = sum(vec) / self.dim
            var = sum((v - mean)**2 for v in vec) / self.dim
            std = (var + self.eps) ** 0.5

            norm = [ (vec[i] - mean) / std for i in range(len(vec)) ]
            out.append([ norm[i] * self.gamma[i] + self.beta[i] for i in range(len(vec)) ])

        return out