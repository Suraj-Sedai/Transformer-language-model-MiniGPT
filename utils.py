import numpy as np
from math import sqrt

def random_matrix(shape):
    rows, cols = shape
    return [[np.random.uniform(-0.01, 0.01) for _ in range(cols)] for _ in range(rows)]

def matmul(vec, mat):
    m = len(vec)
    n = len(mat[0])
    result = [0]*n
    for col in range(n):
        s = 0
        for row in range(m):
            s += vec[row] * mat[row][col]
        result[col] = s
    return result

def add_vectors(a,b):
    return [a[i]+b[i] for i in range(len(a))]

def add_vectors_list(A,B):
    return [add_vectors(A[i],B[i]) for i in range(len(A))]

def relu(x):
    return np.maximum(x,0)

def zeros(n):
    return [0.0]*n

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(probs, target_id):
    return -np.log(probs[target_id] + 1e-8)

def grad_cross_entropy(logits, target_id):
    probs = softmax(logits)
    probs[target_id] -= 1
    return probs
