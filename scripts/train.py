import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tokenizer import BPETokenizer
from model.transformer_model import TransformerModel
from utils import softmax, cross_entropy

# --- Tiny training corpus ---
text = "hello world hello transformer model mini gpt"

# --- Initialize tokenizer ---
tokenizer = BPETokenizer(vocab_size=50)
tokenizer.train(text)

# --- Encode text to token IDs ---
ids = tokenizer.encode(text)  # e.g., [1,5,1,20,...]
seq_len = 4  # context length
lr = 0.01
epochs = 100

# --- Prepare model ---
model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=16,
    num_heads=2,
    num_layers=2,
    ffn_hidden_dim=32
)

# --- Training loop ---
for epoch in range(epochs):
    total_loss = 0

    for i in range(len(ids) - seq_len):
        x_ids = ids[i:i+seq_len]
        y_id = ids[i+seq_len]  # next token

        # Forward pass
        logits, hidden = model.forward_with_hidden(x_ids)
        last_logits = logits[-1]  # shape: (vocab_size,)
        probs = softmax(last_logits)

        # --- Compute loss ---
        loss = cross_entropy(probs, y_id)
        total_loss += loss

        # --- Gradient w.r.t logits ---
        grad = probs.copy()
        grad[y_id] -= 1  # dL/dlogits

        # --- Update output layer (Wo and bo) ---
        # hidden[-1] is the last hidden vector (shape: embed_dim)
        model.Wo = np.array(model.Wo)  # ensure numpy
        model.Wo -= lr * np.outer(hidden[-1], grad)  # shape: (embed_dim, vocab_size)
        model.bo -= lr * grad  # shape: (vocab_size,)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss: {total_loss/len(ids):.4f}")

# --- Generate some text ---
prompt = "hello world"
ids = tokenizer.encode(prompt)
generated_ids = model.generate(ids, max_new_tokens=20, tokenizer=tokenizer)
print(tokenizer.decode(generated_ids))
