from math import sqrt
# tokenizer.py
from collections import Counter, defaultdict
import re

class BPETokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.vocab = {}       # token -> id
        self.inv_vocab = {}   # id -> token
        self.merges = []      # list of merged pairs (a,b)

    def _basic_tokenize_words(self, text: str):
        # simple word split; keep it similar to your current behavior
        # (you can upgrade to regex later)
        return text.lower().split()

    def train(self, text: str):
        text = text.lower()

        # ---- special tokens ----
        for special in ["<unk>", "<pad>"]:
            if special not in self.vocab:
                self.vocab[special] = len(self.vocab)

        words = self._basic_tokenize_words(text)

        # Count unique word frequencies (THIS is the big speedup)
        word_freq = Counter(words)

        # Represent each word as tuple of characters + end marker
        # End marker helps merges not cross word boundary and improves decoding
        def word_to_symbols(w):
            return tuple(list(w) + ["</w>"])

        word_symbols = {w: word_to_symbols(w) for w in word_freq.keys()}

        # Initialize vocab with chars (and </w>)
        for w, sym in word_symbols.items():
            for s in sym:
                if s not in self.vocab:
                    self.vocab[s] = len(self.vocab)

        # ---- BPE loop ----
        while len(self.vocab) < self.vocab_size:
            pair_counts = self._get_pair_frequencies(word_symbols, word_freq)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            a, b = best_pair
            new_token = a + b

            # apply merge to all unique words (not all occurrences)
            word_symbols = self._merge_pair_in_vocab(word_symbols, best_pair)

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)

        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    def _get_pair_frequencies(self, word_symbols, word_freq):
        pair_counts = defaultdict(int)
        for w, symbols in word_symbols.items():
            freq = word_freq[w]
            # count adjacent pairs in this word, weighted by frequency
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i+1])] += freq
        return pair_counts

    def _merge_pair_in_vocab(self, word_symbols, pair_to_merge):
        a, b = pair_to_merge
        merged = a + b
        new_word_symbols = {}

        # merge via single pass through symbols
        for w, symbols in word_symbols.items():
            out = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            new_word_symbols[w] = tuple(out)
        return new_word_symbols

    def encode(self, text: str):
        text = text.lower()
        words = self._basic_tokenize_words(text)
        ids = []

        for w in words:
            symbols = tuple(list(w) + ["</w>"])
            # apply merges in order
            for (a, b) in self.merges:
                symbols = self._merge_pair_symbols(symbols, a, b)
            # map to ids
            for s in symbols:
                ids.append(self.vocab.get(s, self.vocab["<unk>"]))
        return ids

    def _merge_pair_symbols(self, symbols, a, b):
        merged = a + b
        out = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    def decode(self, token_ids):
        # basic decode: join subword tokens then remove </w>
        toks = [self.inv_vocab.get(i, "<unk>") for i in token_ids]
        text = "".join([t.replace("</w>", " ") for t in toks]).strip()
        # cleanup multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text

    
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



