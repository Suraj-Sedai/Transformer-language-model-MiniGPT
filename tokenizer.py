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