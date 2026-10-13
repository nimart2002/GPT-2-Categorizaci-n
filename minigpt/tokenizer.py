"""Tokenizador: convierte tokens (texto) en números enteros y viceversa."""


class Tokenizer:
    def __init__(self, vocab):
        """vocab: lista de tokens. La posición de cada token en la lista es su número."""
        self.vocab = list(vocab)
        assert len(set(self.vocab)) == len(self.vocab), "hay tokens repetidos"
        self.stoi = {token: i for i, token in enumerate(self.vocab)}   # "string to integer"
        self.itos = {i: token for i, token in enumerate(self.vocab)}   # "integer to string"

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, tokens):
        """Lista de tokens -> lista de enteros."""
        return [self.stoi[t] for t in tokens]

    def decode(self, ids):
        """Lista de enteros -> lista de tokens."""
        return [self.itos[i] for i in ids]