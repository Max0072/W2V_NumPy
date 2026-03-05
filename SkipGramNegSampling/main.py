import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def multi_hot(ids, size):
    array = np.zeros(size, dtype=np.float32)
    array[ids] = 1.0
    return array

def tokenize(text):
    return text.lower().replace(".", " ").replace(",", " ").split()

def loss(probs, all_labels):
    return -np.sum(all_labels * np.log(probs) + (1 - all_labels) * np.log(1 - probs))

def build_pairs(token_ids, window):
    pairs = []
    n = len(token_ids)
    for i in range(n):
        context = []
        for j in range(i-window, i+window+1):
            if 0 <= j < n and token_ids[j] != token_ids[i]:
                context.append(token_ids[j])
        if len(context) > 0:
            pairs.append((token_ids[i], context))
    return pairs



class SkipGram:
    def __init__(self, emb_dim=50, random_state=42):
        self.emb_dim = emb_dim
        self.rng = np.random.RandomState(random_state)
        self.vocab_size = 0
        self.word2id = {}
        self.id2word = {}

    def get_negative_samples(self, target_ids, num_negative, rng):
        sampling_probs = self.counts / sum(self.counts)
        negative_ids = rng.choice(self.vocab_size, size=num_negative, p=sampling_probs, replace=True)
        mask = np.isin(negative_ids, target_ids)
        while np.any(mask):
            negative_ids[mask] = rng.choice(self.vocab_size, size=np.sum(mask), p=sampling_probs, replace=True)
            mask = np.isin(negative_ids, target_ids)
        return negative_ids

    def build_vocab(self, words):
        unique_words, self.counts = np.unique(np.array(words), return_counts=True)
        self.vocab_size = len(unique_words)
        self.word2id = {w: i for i, w in enumerate(unique_words)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        return self.word2id, self.id2word


    def backward(self, word_id, word_emb, probs, ids, lbls):
        dlogits = (probs - lbls).astype(np.float32)    #(len(all_ids),)

        dW_out = np.zeros_like(self.W_out)
        dW_out[:, ids] += np.outer(word_emb, dlogits)           # (E, 1) @ (1, len(all_ids)) = (E, len(all_ids))

        dh = self.W_out[:, ids] @ dlogits               # (E, len(all_ids)) @ (len(all_ids),) = (E,)

        dW_in = np.zeros_like(self.W_in)
        dW_in[word_id] += dh

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

            for i, (word_id, target_ids) in enumerate(pairs):
                word_emb = self.W_in[word_id]

                # Negative sampling
                negative_ids = self.get_negative_samples(target_ids, num_negative=4, rng=self.rng)
                all_ids = np.concatenate([target_ids, negative_ids])
                all_labels = np.concatenate([
                    np.ones(len(target_ids)),
                    np.zeros(len(negative_ids))
                ])

                # forward
                logits = word_emb @ self.W_out[:, all_ids]
                probs = sigmoid(logits)

                # Loss, Backward and Update weights
                total_loss += loss(probs, all_labels)
                dW_in, dW_out = self.backward(word_id, word_emb, probs, all_ids, all_labels)
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
    model = SkipGram(100)
    text = "..." # Fill the gap with you own text
    model.fit(text)
    word = "..."  # Fill the gap with the real word
    word_embedding = model.get_embedding(word)


if __name__ == "__main__":
    main()
