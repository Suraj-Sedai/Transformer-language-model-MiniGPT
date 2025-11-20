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
        
