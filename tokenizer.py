from math import sqrt,exp
import random
import numpy as np

'''Some helper functions'''

def random_matrix(shape):
    rows, cols = shape
    return [
        [random.uniform(-0.01, 0.01) for _ in range(cols)]
        for _ in range(rows)
    ]
def matmul(vec, mat):
    # vec: [m]
    # mat: [m][n]
    m = len(vec)
    n = len(mat[0])
    result = [0]*n

    for col in range(n):
        s = 0
        for row in range(m):
            s += vec[row] * mat[row][col]
        result[col] = s

    return result

# v = [1,2,3]
# m = [
#     [1,0],
#     [0,1],
#     [1,1]
# ]
# print(matmul(v, m)) 

def layer_norm(X):
    # X = [seq_len, embed_dim]
    output = []
    for token_vec in X:
        mean = sum(token_vec)/len(token_vec)
        variance = sum((v-mean)**2 for v in token_vec)/len(token_vec)
        std = sqrt(variance + 1e-5)
        normalized = [(v-mean)/std for v in token_vec]
        output.append(normalized)
    return output

def softmax(x):
    e = np.exp(x - np.max(x))  # for numerical stability
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy_loss(logits, target_id):
    probs = softmax(logits)
    return -np.log(probs[target_id] + 1e-9)  # avoid log(0)

def grad_cross_entropy(logits, target_id):
    probs = softmax(logits)
    probs[target_id] -= 1
    return probs  # gradient w.r.t logits

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}      # token -> id
        self.inv_vocab = {}  # id -> token
        self.merges = []     # list of merge rules

    def train(self, text):
        text = text.lower()

        # ---- special tokens ----
        for special in ["<unk>", "<pad>", " "]:
            if special not in self.vocab:
                self.vocab[special] = len(self.vocab)

        # ---- split into words ----
        words = text.strip().split()
        tokens_list = [self._word_to_chars(w) for w in words]

        # ---- add unique chars ----
        for token_list in tokens_list:
            for t in token_list:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)

        # ---- BPE loop ----
        while len(self.vocab) < self.vocab_size:
            pair_counts = self.get_pair_frequencies(tokens_list)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            tokens_list = self._merge_pair(tokens_list, best_pair)

            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            self.merges.append(best_pair)

        # ---- inverse vocab ----
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}


    def _word_to_chars(self, word):
        # turn a word into a list of char
        return list(word)

    def encode(self, text):
        text = text.lower().strip()
        words = text.split()

        # convert each word into list of characters
        tokens_list = [self._word_to_chars(w) for w in words]

        # apply BPE merges
        for merge_pair in self.merges:
            tokens_list = self._merge_pair(tokens_list, merge_pair)

        # convert tokens into ids (safe lookup)
        token_ids = []
        for token_list in tokens_list:
            for token in token_list:
                token_ids.append(self.vocab.get(token, self.vocab["<unk>"]))

        return token_ids


    def decode(self, token_ids):
        tokens = [self.inv_vocab.get(i, "<unk>") for i in token_ids]
        text = ' '.join(tokens)
        return text


    def get_pair_frequencies(self, tokens_list):
        # get frequencies of adjacent token pairs
        pair_counts = dict()
        for token_list in tokens_list:
            for i in range(len(token_list)-1):
                pair = (token_list[i], token_list[i+1])

                if pair not in pair_counts:
                    pair_counts[pair] = 1
                else:
                    pair_counts[pair] +=1
        return pair_counts
    
    def _merge_pair(self, tokens_list,pair_to_merge):
        #pair to merge in tuple
        a = pair_to_merge[0]
        b = pair_to_merge[1]
        new_tokens_list = []
        #processing each word one by one
        for token_list in tokens_list:
            merged_word = []
            i = 0
            while i < len(token_list):
                if i < len(token_list)-1 and token_list[i] == a and token_list[i + 1] == b:
                    #merge two token
                    merged_token = a+b
                    merged_word.append(merged_token)
                    i +=2
                else:
                    merged_word.append(token_list[i])
                    i +=1
            #add processed word back to list
            new_tokens_list.append(merged_word)
        return new_tokens_list
    
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X = []
        self.y = []
    
    def fit(self,X,y):
        self.X = X
        self.y = y

    def euclidean_distance(self,point1, point2):
        #point1 and point2
        sum_of_square = 0
        for i in range(len(point1)):

            #compute the difference between each corresponding features
            diff = (point1[i]-point2[i])
            #square the difference to ensure distance is positive
            squared = diff * diff

            sum_of_square += squared

        distance = sqrt(sum_of_square)
        return distance
    
    def get_k_nearest_neighbors(self, training_data, training_labels, new_point, k):
        distances = []
        for i in range (len(training_data)):
            #get current example from the training dataset
            current_point = training_data[i]

            #calculate distance between new point and training point
            dist = self.euclidean_distance(new_point, current_point)

            #store pair
            distances.append((dist, training_labels[i]))
        
        #SORT DISTANCE
        distances = sorted(distances, key=lambda x: x[0])
        #select first k entries from sorted list
        neighbours = [distances[i] for i in range(k)]

        return neighbours
    
    def majority_vote(self, neighbors):
        label_count = dict()
        for pair in neighbors:
            label = pair[1]
            if label not in label_count:
                label_count[label] = 1
            else:
                label_count[label] += 1
        # Find label with highest count

        most_common_label = max(label_count, key=label_count.get)

        return most_common_label

    def predict(self, test_point):
        distances = []

        for sample, label in zip(self.X, self.y):
            d = self.euclidean_distance(test_point, sample)
            distances.append((d,label))

        #sorting
        distances = sorted(distances, key=lambda x: x[0])
        #take the k nearest neighbour
        k_neighbour = distances[:self.k]
        #use majority vote to predict
        predicted_label = self.majority_vote(k_neighbour)

        return predicted_label

class TokenEmbedding:
    def __init__(self, vocab_size, embed_dim):
        #create embedding matrix with random small values
        self.W = random_matrix(shape=(vocab_size, embed_dim))
    
    def forward(self, token_ids):
        #token_ids : like [ 8,6,4,11]
        embeddings = []
        for id in token_ids:
            #lookup = row from embedding matrix
            vector = self.W[id]
            embeddings.append(vector)
        return embeddings
    
class PosEmbedding:
    def __init__(self,max_len, embed_dim):
        self.P = random_matrix(shape=(max_len, embed_dim))
    def forward(self, token_ids):
        length = len(token_ids)
        #add position bector to each token embedding
        return [self.P[pos] for pos in range(length)]



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
def zeros(n):
    return [0.0 for _ in range(n)]

def relu(x):
    return np.maximum(x, 0)


def add_vectors(a, b):
    """Element-wise add two vectors (same length)."""
    return [a[i] + b[i] for i in range(len(a))]
def add_vectors_list(A, B):
    return [add_vectors(A[i], B[i]) for i in range(len(A))]
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

class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)

    def forward(self, X):
        # if batch dim exists, drop it for simplicity
        if isinstance(X, np.ndarray) and X.ndim == 3:
            X = X[0].tolist()  # shape -> [seq_len, embed_dim]

        # LayerNorm + MultiHeadAttention
        normed = self.ln1.forward(X)
        attn_out = self.mha.forward(normed)  # shape: [seq_len, embed_dim]

        # Residual
        x2 = add_vectors_list(X, attn_out)

        # LayerNorm + FeedForward
        normed2 = self.ln2.forward(x2)
        ff_out = self.ff.forward(normed2)

        # Residual
        out = add_vectors_list(x2, ff_out)
        return out

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

# Example tiny corpus
text = "hello world hello transformer model mini gpt"

# Initialize and train tokenizer
tokenizer = BPETokenizer(vocab_size=50)
tokenizer.train(text)

# Convert text to token IDs
token_ids = tokenizer.encode(text)  # e.g., [1, 5, 1, 20, 3, ...]
seq_len = 4  # number of tokens in input
X_train = []
y_train = []

for i in range(len(token_ids) - seq_len):
    X_train.append(token_ids[i:i+seq_len])
    y_train.append(token_ids[i+seq_len])

X_train = np.array(X_train)  # shape: (num_samples, seq_len)
y_train = np.array(y_train)  # shape: (num_samples,)

lr = 0.01  # learning rate
model = TransformerModel(vocab_size=tokenizer.vocab_size, embed_dim=16,
                         num_heads=2, num_layers=2, ffn_hidden_dim=32)

epochs = 100
import numpy as np

# --- Hyperparameters ---
lr = 0.1        # learning rate
epochs = 200
seq_len = 5     # small sequence length for tiny corpus

# Assume you already have:
# model: TransformerModel instance
# tokenizer: BPETokenizer instance
# text: training text (string)

# Encode the text into token IDs
ids = tokenizer.encode(text)

# Training loop (simple next-token prediction)
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(len(ids) - seq_len):
        x_ids = ids[i:i+seq_len]
        y_id = ids[i+seq_len]

        logits = model.forward(x_ids)
        pred = logits[-1]  # last token logits

        # softmax & loss
        exp_pred = np.exp(pred - np.max(pred))
        probs = exp_pred / np.sum(exp_pred)
        loss = -np.log(probs[y_id] + 1e-8)
        total_loss += loss

        # gradient for Wo & bo
        grad = probs.copy()
        grad[y_id] -= 1.0

        # last hidden vector
        X_block = model.blocks[-1].forward(
            [add_vectors(model.token_embed.forward(x_ids)[-1],
                         model.pos_embed.forward(x_ids)[-1])]
        )
        h = np.array(X_block[-1])

        # update
        model.Wo -= lr * np.outer(h, grad)
        model.bo -= lr * grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss: {total_loss / len(ids):.4f}")


prompt = "hello world"
ids = tokenizer.encode(prompt)
generated_ids = model.generate(ids, max_new_tokens=20, tokenizer=tokenizer)
print(tokenizer.decode(generated_ids))



'''Test cases for all the clasees and functions'''
# if __name__ == "__main__":
        
#     X = [
#         [1, 2],
#         [2, 3],
#         [3, 3],
#         [8, 7],
#         [9, 8],
#         [10, 8]
#     ]

#     y = ["A", "A", "A", "B", "B", "B"]

#     knn = KNN(k=3)
#     knn.fit(X, y)

#     print(knn.predict([2, 2]))   # should give "A"
#     print(knn.predict([9, 7]))   # should give "B"


#     text = "this is a test. this test is fun."
#     tokenizer = BPETokenizer(vocab_size=50)
#     tokenizer.train(text)
#     print(tokenizer.vocab)
#     print(tokenizer.merges)

#     text = "this is a test"
#     tokenizer = BPETokenizer(vocab_size=50)
#     tokenizer.train(text)

#     ids = tokenizer.encode("this is a test")
#     print(ids)

#     decoded_text = tokenizer.decode(ids)
#     # print(decoded_text)



#     def add_vectors(a,b):
#         return [a[i] + b[i] for i in range(len(a))]
    
#     token_embed = TokenEmbedding(vocab_size=1000, embed_dim=32)
#     pos_embed = PosEmbedding(max_len=512, embed_dim=32)

#     ids = [8,6,4,11]

#     token_vectors = token_embed.forward(ids)
#     pos_vectors   = pos_embed.forward(ids)

#     final_vectors = [
#         add_vectors(token_vectors[i], pos_vectors[i])
#         for i in range(len(ids))
#     ]

#     print(final_vectors)
    
#     #Fake tiny example
#     embed_dim = 4
#     X = [
#         [0.1, 0.2, 0.3, 0.4],
#         [0.5, 0.4, 0.3, 0.2]
#     ]

#     att = SelfAttention(embed_dim)
#     Q, K, V = att.forward(X)

#     print("Q:", Q)
#     print("K:", K)
#     print("V:", V)

    # # ---- BUILD INPUT X ----
    # tokenizer = BPETokenizer(vocab_size=1000)
    # tokenizer.train("this is a test corpus for building tiny gpt tokenizer")
    # token_embedding = TokenEmbedding(vocab_size=1000, embed_dim=4)
    # pos_embedding   = PosEmbedding(max_len=50, embed_dim=4)

    # ids = tokenizer.encode("this is a test")

    # token_embed = token_embedding.forward(ids)
    # pos_embed = pos_embedding.forward(ids)

    # # Add token+pos embeddings
    # X = []
    # for i in range(len(token_embed)):
    #     vec = []
    #     for a, b in zip(token_embed[i], pos_embed[i]):
    #         vec.append(a + b)
    #     X.append(vec)

    # # ---- RUN TRANSFORMER BLOCK ----
    # tb = TransformerBlock(embed_dim=4, num_heads=2, ffn_hidden_dim=16)
    # out = tb.forward(X)

    # print("block out shape:", len(out), len(out[0]))
    # print(out)


    # print("block out shape:", len(out), len(out[0]))
    # print(out)


    # print("Testing full TransformerBlock...")
    # tb = TransformerBlock(embed_dim=4, num_heads=2, ff_hidden_dim=16)

    # out = tb.forward(X)
    # print("block out shape:", len(out), len(out[0]))
    # print(out)

    # print("mha out shape:", len(out), len(out[0]))  # expect seq_len x embed_dim

    # small test X (seq_len=4, embed_dim must match your token/embed dims)


    # embed_dim = 4
    # num_heads = 2  # head_dim = 2
    # token_embedding = TokenEmbedding(vocab_size=100, embed_dim=embed_dim)
    # pos_embedding = PosEmbedding(max_len=20, embed_dim=embed_dim)
    # tokenizer = BPETokenizer(vocab_size=100)
    # tokenizer.train("this is a test")   # small corpus ok

    # ids = tokenizer.encode("this is a test")
    # token_vectors = token_embedding.forward(ids)
    # pos_vectors   = pos_embedding.forward(ids)
    # X = [[a+b for a,b in zip(token_vectors[i], pos_vectors[i])] for i in range(len(ids))]

    # mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    # out = mha.forward(X)
    # tb = TransformerBlock(embed_dim=4, num_heads=2, ff_hidden_dim=16)
    # out = tb.forward(X)

    # model = TransformerModel(vocab_size=200, max_len=32, embed_dim=8, num_heads=2, num_layers=2, ffn_hidden_dim=32)
    # tokenizer = BPETokenizer(vocab_size=200)
    # tokenizer.train("this is a tiny corpus for testing")
    # ids = tokenizer.encode("this is a test")
    # logits = model.forward(ids)
    # print("logits shape:", len(logits), len(logits[0]))   # expect seq_len x vocab_size
    
    # training_text = """
    # hello world this is a tiny training dataset
    # hello there how are you
    # i am building a tiny transformer language model
    # """

    # tokenizer = BPETokenizer(vocab_size=1000)
    # tokenizer.train(training_text)

    # # ACTUAL vocab size after training
    # vocab_size = len(tokenizer.vocab)

    # model = TransformerModel(vocab_size, embed_dim=64)

    # prompt = "hello"
    # ids = tokenizer.encode(prompt)

    # generated = model.generate(ids, max_new_tokens=20, tokenizer=tokenizer)

    # print(tokenizer.decode(generated))