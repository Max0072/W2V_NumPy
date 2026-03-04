import numpy as np


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

def one_hot(idx, size):
    array = np.zeros(size, dtype=np.float32)
    array[idx] = 1.0
    return array

def tokenize(text):
    return text.lower().replace(".", " ").replace(",", " ").split()

def loss(probs, target_id):
    return -np.log(probs[target_id])

def build_pairs(token_ids, window):
    pairs = []
    n = len(token_ids)
    for i in range(n):
        context = []
        for j in range(i-window, i+window+1):
            if 0 <= j < n and token_ids[j] != token_ids[i]:
                context.append(token_ids[j])
        if len(context) > 0:
            pairs.append((context, token_ids[i]))
    return pairs


class ContinuousBagOfWords:
    def __init__(self, emb_dim=50, random_state=42):
        self.emb_dim = emb_dim
        self.rng = np.random.RandomState(random_state)
        self.vocab_size = 0
        self.word2id = {}
        self.id2word = {}

    def build_vocab(self, words):
        unique_words = set(words)
        self.vocab_size = len(unique_words)
        self.word2id = {w: i for i, w in enumerate(unique_words)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        return self.word2id, self.id2word

    def forward(self, token_ids):
        token_embeddings = self.W_in[token_ids]
        h = token_embeddings.mean(axis=0)
        logits = h @ self.W_out
        probs = softmax(logits)
        info = (token_ids, h, probs)
        return probs, info

    def backward(self, info, target_id):
        token_ids, h, probs = info
        y = one_hot(target_id, self.vocab_size)     #(V,)

        dlogits = (probs - y).astype(np.float32)    #(V,)

        dW_out = np.outer(h, dlogits)           # (E, 1) @ (1, V) = (E, V)

        dh = self.W_out @ dlogits               # (E, V) @ (V,) = (E,)

        dW_in = np.zeros_like(self.W_in)
        grad_each = dh / len(token_ids)
        for token_id in token_ids:
            dW_in[token_id] += grad_each

        return dW_in, dW_out

    def step(self, dW_in, dW_out, lr):
        self.W_in -= lr * dW_in
        self.W_out -= lr * dW_out

    def fit(self, corpus, epochs=50, window=2, lr=0.05):
        words = tokenize(corpus)
        self.build_vocab(words)

        token_ids = []
        for word in words:
            token_ids.append(self.word2id[word])
        pairs = build_pairs(token_ids, window=window)

        self.W_in = self.rng.normal(0, 0.01, size=(self.vocab_size, self.emb_dim)).astype(np.float32)
        self.W_out = self.rng.normal(0, 0.01, size=(self.emb_dim, self.vocab_size)).astype(np.float32)


        for epoch in range(epochs):
            self.rng.shuffle(pairs)
            total_loss = 0.0

            for i, (context, target_id) in enumerate(pairs):
                probs, info = self.forward(context)
                total_loss += loss(probs, target_id)
                dW_in, dW_out = self.backward(info, target_id)
                self.step(dW_in, dW_out, lr=lr)

            print(f"Epoch: {epoch} | Loss: {total_loss / len(pairs)}")


    def get_embedding(self, word):
        word = word.lower()
        if word not in self.word2id:
            print("Haven't seen this word")
            return None
        word_id = self.word2id[word]
        return self.W_in[word_id]


def main():
    model = ContinuousBagOfWords(100)
    text = "..." # Fill the gap with you own tex
    model.fit(text)
    word = "..."  # Fill the gap with the real word
    word_embedding = model.get_embedding(word)
    print(word_embedding)


if __name__ == "__main__":
    main()


