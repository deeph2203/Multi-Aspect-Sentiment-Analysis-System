# src/features/embeddings.py - GloVe loader and custom Word2Vec (SGNS) implementation
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_glove_matrix(glove_path, corpus_words, train_words, dim=100):
    """
    Loads pretrained GloVe embeddings for words present in the corpus.
    Words in train set are added to vocabulary; validation/test words are added
    only if they exist in GloVe (preventing label leakage).
    """
    glove = {}
    glove_path = Path(glove_path)
    if glove_path.exists():
        with open(glove_path, encoding='utf8') as f:
            for line in f:
                w, *vec = line.rstrip().split(' ')
                if w in corpus_words and len(vec) == dim:
                    glove[w] = np.asarray(vec, dtype='float32')

    vocab = {'<pad>': 0, '<unk>': 1}
    for w in train_words:
        if w not in vocab:
            vocab[w] = len(vocab)
            
    for w in corpus_words:
        if w not in vocab and w in glove:
            vocab[w] = len(vocab)

    emb_matrix = np.random.normal(0, 0.1, (len(vocab), dim)).astype('float32')
    emb_matrix[0] = 0  # padding token is all zeros
    for w, idx in vocab.items():
        if w in glove:
            emb_matrix[idx] = glove[w]

    return vocab, emb_matrix, glove


def train_sgns(sentences, dim=50, window=5, neg=5, epochs=15, min_count=2, lr=0.01, bs=512):
    """
    Trains a Skip-gram with Negative Sampling (SGNS) Word2Vec model on local corpus sentences.
    """
    cnt = Counter(w for s in sentences for w in s)
    vocab = [w for w, c in cnt.items() if c >= min_count]
    w2i = {w: i for i, w in enumerate(vocab)}

    cen, con = [], []
    for s in sentences:
        ids = [w2i[w] for w in s if w in w2i]
        for i, c in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    cen.append(c)
                    con.append(ids[j])

    if not cen:
        return vocab, w2i, torch.zeros(len(vocab), dim)

    cen, con = torch.tensor(cen), torch.tensor(con)
    freq = torch.tensor([cnt[w] for w in vocab], dtype=torch.float) ** 0.75
    freq = freq / freq.sum()

    e_in = nn.Embedding(len(vocab), dim)
    e_out = nn.Embedding(len(vocab), dim)
    nn.init.uniform_(e_in.weight, -0.5 / dim, 0.5 / dim)
    nn.init.zeros_(e_out.weight)

    opt = torch.optim.Adam(list(e_in.parameters()) + list(e_out.parameters()), lr=lr)

    for _ in range(epochs):
        perm = torch.randperm(len(cen))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            v = e_in(cen[b])
            u = e_out(con[b])
            n_samples = torch.multinomial(freq, len(b) * neg, replacement=True).view(len(b), neg)
            n = e_out(n_samples)
            loss = -(F.logsigmoid((v * u).sum(1)) + F.logsigmoid(-(n @ v.unsqueeze(2)).squeeze(2)).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    return vocab, w2i, e_in.weight.detach()


def most_similar(word, vocab, w2i, matrix, topn=5):
    """Finds top-n cosine similarity neighbors for a word."""
    if word not in w2i:
        return ['(not in vocabulary)']
    m_norm = F.normalize(matrix, dim=1)
    sims = m_norm @ m_norm[w2i[word]]
    return [vocab[i] for i in sims.argsort(descending=True)[1:topn + 1].tolist()]
