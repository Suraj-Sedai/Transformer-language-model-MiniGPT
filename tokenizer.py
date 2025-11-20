class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}      # token -> id
        self.inv_vocab = {}  # id -> token
        self.merges = []     # list of merge rules

    def train(self, text):
        # will implement BPE training here
        pass

    def encode(self, text):
        # string -> token ids
        pass

    def decode(self, token_ids):
        # token ids -> string
        pass
