import numpy as np

def softmax(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / np.sum(ex, axis=-1, keepdims=True)

class Attention:
    def __init__(self):
        pass

    def forward(self, Q, K, V):
        dk = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(dk)

        weights = softmax(scores)
        output = np.matmul(weights, V)

        return output, weights
