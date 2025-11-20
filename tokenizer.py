from math import sqrt
class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}      # token -> id
        self.inv_vocab = {}  # id -> token
        self.merges = []     # list of merge rules

    def train(self, text):
        # will implement BPE training here
        text.lower()
        word = text.strip().split()
        tokens_list = [self._word_to_chars(w) for w in word]

    def _word_to_chars(self, word):
        # turn a word into a list of char
        return list(word)

    def encode(self, text):
        # string -> token ids
        pass

    def decode(self, token_ids):
        # token ids -> string
        pass

    def get_pair_frequencies(self, tokens_list):
        # get frequencies of adjacent token pairs
        pair_counts = dict()
        for token_list in tokens_list:
            for i in range(len(token_list-2)):
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
        for i in range (len(training_data)-1):
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
    
X = [
    [1, 2],
    [2, 3],
    [3, 3],
    [8, 7],
    [9, 8],
    [10, 8]
]

y = ["A", "A", "A", "B", "B", "B"]

knn = KNN(k=3)
knn.fit(X, y)

print(knn.predict([2, 2]))   # should give "A"
print(knn.predict([9, 7]))   # should give "B"
