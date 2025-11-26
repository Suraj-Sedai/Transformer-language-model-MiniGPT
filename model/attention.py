import numpy as np
from math import sqrt
from utils import matmul, random_matrix



'''Implement Q, K, V Weight Matrices + Compute Q, K, V vectors''' 
class SelfAttention:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim

    def forward(self, Q, K, V):
        """
        Q, K, V shapes: (batch, seq_len, embed_dim)
        Return:
            output: (batch, seq_len, embed_dim)
            weights: (batch, seq_len, seq_len)
        """

        B, T, D = Q.shape

        # 1. Compute attention scores = Q·K^T
        # shape -> (B, T, T)
        scores = np.matmul(Q, K.transpose(0, 2, 1))

        # 2. Scale
        scores = scores / np.sqrt(D)

        # 3. Softmax across last dimension
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # 4. Weighted sum over V
        # (B, T, T) @ (B, T, D) -> (B, T, D)
        output = np.matmul(weights, V)

        return output, weights

    def dot(self,a,b):
        sum = 0
        for i in range(len(a)):
            sum += a[i] * b[i]
        return sum
    
    def soft_max(self, scores):
        scores = np.array(scores, dtype=np.float64)   # convert to array
        scores = scores - np.max(scores)             # numerical stability
        exps = np.exp(scores)                         # numpy exp works on arrays
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        return probs

    def attention_scores(self, Q, K):
        matrix = []
        for qi in Q:
            row = []
            for kj in K:
                row.append(self.dot(qi, kj))
            matrix.append(row)
        return matrix
    
    def weighted_sum(self, weights, V):
        dim = len(V[0])
        result = [0.0 for _ in range(dim)]
        for i in range(len(V)):
            for d in range(dim):
                result[d] += weights[i] * V[i][d]
        return result
    
    def compute_attention(self,Q,K,V):
        scores = self.attention_scores(Q,K)
        # scale
        for i in range(len(scores)):
            for j in range(len(scores[i])):
                scores[i][j] /= sqrt(self.embed_dim)
        output = []
        for row in scores:
            w = self.soft_max(row)
            out_vec = self.weighted_sum(w, V)
            output.append(out_vec)

        return output
class AttentionHead:
    def __init__(self, embed_dim, head_dim):
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.W_q = random_matrix((embed_dim, head_dim))
        self.W_k = random_matrix((embed_dim, head_dim))
        self.W_v = random_matrix((embed_dim, head_dim))

    # -------------------------
    # Copy these 5 methods below
    # -------------------------
    def dot(self, a, b):
        s = 0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def attention_scores(self, Q, K):
        matrix = []
        for qi in Q:
            row = []
            for kj in K:
                row.append(self.dot(qi, kj))
            matrix.append(row)
        return matrix

    def soft_max(self, scores):
        scores = np.array(scores, dtype=np.float64)   # convert to array
        scores = scores - np.max(scores)             # numerical stability
        exps = np.exp(scores)                         # numpy exp works on arrays
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        return probs

    def weighted_sum(self, weights, V):
        dim = len(V[0])
        result = [0.0 for _ in range(dim)]

        for i in range(len(V)):
            for d in range(dim):
                result[d] += weights[i] * V[i][d]
        return result

    def compute_attention(self, Q, K, V):
        scores = self.attention_scores(Q, K)
        for i in range(len(scores)):
            for j in range(len(scores[i])):
                scores[i][j] /= sqrt(self.head_dim)

        output = []
        for row in scores:
            w = self.soft_max(row)
            out_vec = self.weighted_sum(w, V)
            output.append(out_vec)
        return output

    # -------------------------
    # Main forward
    # -------------------------
    def forward(self, X):
        Q, K, V = [], [], []
        for token_vec in X:
            Q.append(matmul(token_vec, self.W_q))
            K.append(matmul(token_vec, self.W_k))
            V.append(matmul(token_vec, self.W_v))

        return self.compute_attention(Q, K, V)

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.heads = [ AttentionHead(embed_dim, self.head_dim) for _ in range(num_heads) ]

        # 🔥 Final linear layer after concatenation
        self.Wo = random_matrix((embed_dim, embed_dim))

    def concat_along_last_dim(self, head_outputs):
        # head_outputs: list of [seq_len x head_dim] arrays
        seq_len = len(head_outputs[0])
        num_heads = len(head_outputs)
        head_dim = len(head_outputs[0][0])

        # initialize result
        result = []

        # loop over tokens
        for i in range(seq_len):
            concatenated = []
            for head in head_outputs:
                concatenated.extend(head[i])  # append head vector for token i
            result.append(concatenated)

        return result

    def forward(self, X):
        # X shape: [seq_len, embed_dim]

        head_outputs = []

        for head in self.heads:
            # for each head, run attention
            # BUT: X must first be projected down to head_dim

            # Create Q,K,V using head weight matrices
            # (selfattention already does this)

            out = head.forward(X)  
            # shape: [seq_len, head_dim]

            head_outputs.append(out)

        # concatenate outputs from all heads
        # final shape: [seq_len, embed_dim]
        concatenated = self.concat_along_last_dim(head_outputs)

        # final linear projection
        final_output = []
        for vec in concatenated:
            final_output.append(matmul(vec, self.Wo))

        return final_output