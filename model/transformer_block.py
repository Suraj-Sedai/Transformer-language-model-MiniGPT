from utils import add_vectors_list
from model.attention import MultiHeadAttention
from model.layers import LayerNorm, FeedForward

class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)

    def forward(self, X):
        normed = self.ln1.forward(X)
        attn_out = self.mha.forward(normed)
        x2 = add_vectors_list(X, attn_out)
        normed2 = self.ln2.forward(x2)
        ff_out = self.ff.forward(normed2)
        return add_vectors_list(x2, ff_out)
