import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tokenizer import BPETokenizer
from model.transformer_model import TransformerModel
from utils import softmax, cross_entropy

# --- Better training corpus ---
text = """
Artificial Intelligence (AI) allows machines to perform tasks that typically require human intelligence. 
Machine Learning (ML) enables computers to learn patterns from data. 
Deep Learning (DL) uses neural networks to analyze large datasets and improve performance over time. 
AI applications include language understanding, image recognition, and decision making. 
Reinforcement Learning trains agents to take actions to maximize rewards. 
Natural Language Processing allows machines to understand and generate human language. 
Computer Vision enables image and video analysis. 
Generative AI can create text, images, and music automatically.
"""

# --- Initialize tokenizer ---
tokenizer = BPETokenizer(vocab_size=500)
tokenizer.train(text)
ids = tokenizer.encode(text)

# --- Hyperparameters ---
seq_len = 6       # context length
lr = 0.01
epochs = 100
top_k = 5         # for sampling diversity

# --- Initialize model ---
model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=2,
    num_layers=4,
    ffn_hidden_dim=256
)

# --- Training loop ---
for epoch in range(epochs):
    total_loss = 0

    for i in range(len(ids) - seq_len):
        x_ids = ids[i:i+seq_len]
        y_id = ids[i+seq_len]

        # Forward pass
        logits, hidden = model.forward_with_hidden(x_ids)
        last_logits = logits[-1]
        probs = softmax(last_logits)

        # Loss
        loss = cross_entropy(probs, y_id)
        total_loss += loss

        # Gradients w.r.t logits
        grad = probs.copy()
        grad[y_id] -= 1

        # Update output layer
        model.Wo -= lr * np.outer(hidden[-1], grad)
        model.bo -= lr * grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss: {total_loss/len(ids):.4f}")

# --- Text generation ---
def sample_next_token(logits, top_k=5):
    # Top-k sampling
    idxs = np.argsort(logits)[-top_k:]
    probs = softmax(logits[idxs])
    return np.random.choice(idxs, p=probs)

prompt = "Artificial Intelligence enables"
ids = tokenizer.encode(prompt)
generated_ids = ids.copy()

for _ in range(30):
    logits, hidden = model.forward_with_hidden(generated_ids[-seq_len:])
    next_id = sample_next_token(logits[-1], top_k=top_k)
    generated_ids.append(next_id)

print("Generated text:")
print(tokenizer.decode(generated_ids))
