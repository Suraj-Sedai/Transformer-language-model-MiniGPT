from math import sqrt,exp
import random

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

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}      # token -> id
        self.inv_vocab = {}  # id -> token
        self.merges = []     # list of merge rules

    def train(self, text):
        # will implement BPE training here
        text = text.lower()
        word = text.strip().split()
        tokens_list = [self._word_to_chars(w) for w in word]

        #initialize vocab with unique char
        for token_list in tokens_list:
            for token in token_list:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        
        while len(self.vocab) < self.vocab_size:
            #count frequency of adjacent paire
            pair_counts = self.get_pair_frequencies(tokens_list)
            #if no pair found stop
            if not pair_counts:
                break

            #find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            #merge this pair in all words
            tokens_list = self._merge_pair(tokens_list, best_pair)
            #add merged token in vocabulary
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
            #save merge rule for encoding later
            self.merges.append(best_pair)
        #build inverse vocal id. token for decending
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}


    def _word_to_chars(self, word):
        # turn a word into a list of char
        return list(word)

    def encode(self, text):
        # string -> token ids
        text = text.lower().strip()
        words = text.split()

        #convert word to list of characters
        tokens_list = [self._word_to_chars(w)for w in words]

        #apply merge rule in order
        for merge_pair in self.merges:
            tokens_list = self._merge_pair(tokens_list, merge_pair)
        
        #convert tokens into ids using vocab
        token_ids = []
        for token_list in tokens_list:
            for token in token_list:
                token_ids.append(self.vocab[token])
        
        return token_ids

    def decode(self, token_ids):
        # token ids -> string
        tokens = [self.inv_vocab[i] for i in token_ids]
        text = ''.join(tokens)
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

        # Weight matrices: [embed_dim x embed_dim]
        self.W_q = random_matrix((embed_dim, embed_dim))
        self.W_k = random_matrix((embed_dim, embed_dim))
        self.W_v = random_matrix((embed_dim, embed_dim))

    def forward(self, X):
        """
        X = list of token vectors, shape: [seq_len, embed_dim]
        returns Q, K, V each of shape: [seq_len, embed_dim]
        """

        Q = []
        K = []
        V = []
        

        for token_vec in X:
            print("token_vec length =", len(token_vec))
            print("W_q rows =", len(self.W_q))
            print("W_q cols =", len(self.W_q[0]))
            q = matmul(token_vec, self.W_q)
            k = matmul(token_vec, self.W_k)
            v = matmul(token_vec, self.W_v)

            Q.append(q)
            K.append(k)
            V.append(v)

        return self.compute_attention(Q, K, V)

    
    def dot(self,a,b):
        sum = 0
        for i in range(len(a)):
            sum += a[i] * b[i]
        return sum
    
    def soft_max(self, scores):
        exps = []
        for s in scores:
            exps.append(exp(s))
        total = sum(exps)
        probs = []
        for e in exps:
            probs.append(e/total)
        
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
        exps = []
        for s in scores:
            exps.append(exp(s))
        total = sum(exps)
        probs = []
        for e in exps:
            probs.append(e / total)
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

        self.heads = []
        for _ in range(num_heads):
            self.heads.append(AttentionHead(embed_dim, self.head_dim))

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

def relu(vec):
    return [max(0, v) for v in vec]

def add_vectors(a, b):
    """Element-wise add two vectors (same length)."""
    return [a[i] + b[i] for i in range(len(a))]

class FeedForward:
    def __init__(self, embed_dim, hidden_dim):
        # correct shapes
        self.W1 = random_matrix((embed_dim, hidden_dim))      # [embed_dim x hidden_dim]
        self.b1 = zeros(hidden_dim)

        self.W2 = random_matrix((hidden_dim, embed_dim))      # [hidden_dim x embed_dim]
        self.b2 = zeros(embed_dim)

    def forward(self, X):
        out = []
        for token_vec in X:

            # First linear layer: X @ W1 + b1
            hidden = matmul(token_vec, self.W1)
            hidden = [hidden[i] + self.b1[i] for i in range(len(hidden))]

            # ReLU activation
            hidden = relu(hidden)

            # Second linear layer
            output_vec = matmul(hidden, self.W2)
            output_vec = [output_vec[i] + self.b2[i] for i in range(len(output_vec))]

            out.append(output_vec)

        return out


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim):
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ffn_hidden_dim)
        self.embed_dim = embed_dim

    def forward(self, X):
        # Multi-head attention
        attn_output = self.mha.forward(X)

        # Residual + LayerNorm
        X = [add_vectors(X[i], attn_output[i]) for i in range(len(X))]
        X = layer_norm(X)

        # Feed-forward
        ffn_output = self.ffn.forward(X)

        # Second residual + layer norm
        X = [add_vectors(X[i], ffn_output[i]) for i in range(len(X))]
        X = layer_norm(X)

        return X


# ---- BUILD INPUT X ----
tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.train("this is a test corpus for building tiny gpt tokenizer")
token_embedding = TokenEmbedding(vocab_size=1000, embed_dim=4)
pos_embedding   = PosEmbedding(max_len=50, embed_dim=4)

ids = tokenizer.encode("this is a test")

token_embed = token_embedding.forward(ids)
pos_embed = pos_embedding.forward(ids)

# Add token+pos embeddings
X = []
for i in range(len(token_embed)):
    vec = []
    for a, b in zip(token_embed[i], pos_embed[i]):
        vec.append(a + b)
    X.append(vec)

# ---- RUN TRANSFORMER BLOCK ----
tb = TransformerBlock(embed_dim=4, num_heads=2, ffn_hidden_dim=16)
out = tb.forward(X)

print("block out shape:", len(out), len(out[0]))
print(out)



# '''Test cases for all the clasees and functions'''
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

