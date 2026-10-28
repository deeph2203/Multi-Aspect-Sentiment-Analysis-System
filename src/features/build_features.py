# src/features/build_features.py - Feature engineering pipeline for TF-IDF and Neural Encodings
from collections import Counter
import json
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import PROCESSED_DIR, GLOVE_FILE, K, MAX_LEN, MAX_ASP, MAX_CTX, LABEL2ID
from src.features.context_window import aspect_idx, window_tokens, opinion_words
from src.features.embeddings import load_glove_matrix


def pad_sequence(seq, target_len, pad_value=0):
    """Pads or truncates a sequence to target_len."""
    return (list(seq) + [pad_value] * target_len)[:target_len]


def build_all_features(nlp=None):
    """
    Executes the entire feature extraction workflow:
    1. Parses aspect-context windows & dependency opinion words
    2. Computes TF-IDF representations (fitted on train only)
    3. Loads GloVe embeddings & builds vocabulary
    4. Packages neural sequence arrays (sent_ids, asp_ids, asp_start, asp_end, ctx_ids)
    5. Persists artifacts to data/processed/
    """
    if nlp is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner'])

    pairs = pd.read_csv(PROCESSED_DIR / 'pairs_prep.csv', dtype={'sentence_id': str})
    pairs['tokens'] = pairs['tokens'].fillna('')
    sents = pd.read_pickle(PROCESSED_DIR / 'sents_prep.pkl')

    docs = dict(zip(sents.sentence_id, nlp.pipe(sents.text.tolist(), batch_size=64)))
    stop_words = nlp.Defaults.stop_words

    # 1. Aspect Context Windows & Opinions
    idx_list, win_list, op_list = [], [], []
    for r in pairs.itertuples():
        doc = docs[r.sentence_id]
        idx = aspect_idx(doc, r.start, r.end)
        if not idx:
            idx = [0]
        idx_list.append(idx)
        win_list.append(' '.join(window_tokens(doc, idx, k=K, stop_words=stop_words)))
        op_list.append(opinion_words(doc, idx))

    pairs['ctx_window'] = win_list
    pairs['opinion'] = [' '.join(o) for o in op_list]
    pairs['ctx'] = [w + ' ' + ' '.join('OP_' + x for x in o) for w, o in zip(win_list, op_list)]

    # 2. GloVe Vocabulary & Embedding Matrix
    sent_toks = {sid: [t.lower_ for t in d] for sid, d in docs.items()}
    split_of = dict(zip(sents.sentence_id, sents.split))
    cnt_split = {
        s: Counter(w for sid, ts in sent_toks.items() if split_of[sid] == s for w in ts)
        for s in ('train', 'val', 'test')
    }
    corpus_words = set().union(*[set(c) for c in cnt_split.values()])
    train_words = [w for w, _ in cnt_split['train'].most_common()]

    vocab, emb_matrix, _ = load_glove_matrix(GLOVE_FILE, corpus_words, train_words, dim=100)

    # 3. Neural Sequence Encodings
    enc = dict(sent_ids=[], sent_len=[], asp_ids=[], asp_start=[], asp_end=[], ctx_ids=[])
    for r, idx in zip(pairs.itertuples(), idx_list):
        toks = sent_toks[r.sentence_id]
        n = len(toks)
        s0 = 0
        if n > MAX_LEN:
            s0 = max(0, min(idx[0] - 10, n - MAX_LEN))
        sent = toks[s0:s0 + MAX_LEN]

        enc['sent_ids'].append(pad_sequence([vocab.get(w, 1) for w in sent], MAX_LEN))
        enc['sent_len'].append(len(sent))
        enc['asp_start'].append(min(max(idx[0] - s0, 0), len(sent) - 1))
        enc['asp_end'].append(min(idx[-1] - s0 + 1, len(sent)))
        enc['asp_ids'].append(pad_sequence([vocab.get(toks[i], 1) for i in idx if i < len(toks)], MAX_ASP))

        lo, hi = max(idx[0] - K, 0), min(idx[-1] + K + 1, n)
        enc['ctx_ids'].append(pad_sequence([vocab.get(w, 1) for w in toks[lo:hi]], MAX_CTX))

    y = pairs.polarity.map(LABEL2ID).values
    arr = {k: np.array(v, dtype=np.int64) for k, v in enc.items()}
    arr['y'] = y.astype(np.int64)
    arr['split'] = pairs.split.values.astype(str)

    # Save processed features
    np.savez_compressed(PROCESSED_DIR / 'features.npz', **arr)
    np.save(PROCESSED_DIR / 'embedding_matrix.npy', emb_matrix)
    (PROCESSED_DIR / 'vocab.json').write_text(json.dumps(vocab))
    pairs.to_csv(PROCESSED_DIR / 'pairs_features.csv', index=False)

    return arr, emb_matrix, vocab, pairs
