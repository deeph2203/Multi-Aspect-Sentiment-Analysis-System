# src/data/preprocessor.py - Preprocessing, lemmatization, and sentence-level splitting
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

KEEP_WORDS = {'no', 'not', 'nor', 'never', "n't", 'but', 'however', 'very', 'too', 'only', 'without'}


def clean_text(text):
    """Removes URLs, HTML tags, and redundant whitespace."""
    if not isinstance(text, str):
        return ""
    t = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    t = re.sub(r'<[^>]+>', ' ', t)
    return ' '.join(t.split())


def to_tokens(doc, stop_words):
    """
    Lemmatizes tokens, converts to lowercase, and removes stopwords
    while strictly preserving negation and contrastive discourse markers.
    """
    return [
        t.lemma_.lower() for t in doc
        if (t.is_alpha or t.text == "n't") and (t.lower_ in KEEP_WORDS or t.lower_ not in stop_words)
    ]


tokenize_and_lemmatize = to_tokens


def split_dataset_by_sentence(sents_df, pairs_df, search_seeds=200):
    """
    Splits the dataset into 70% Train, 15% Validation, and 15% Test strictly by sentence_id
    to ensure sentences with multiple aspects never leak across partitions.
    Finds the optimal seed minimizing class-ratio deviation.
    """
    ids = sents_df.sentence_id.tolist()
    overall = pairs_df.polarity.value_counts(normalize=True)

    def make_split(seed):
        tr, rest = train_test_split(ids, test_size=0.30, random_state=seed)
        va, te = train_test_split(rest, test_size=0.50, random_state=seed)
        return tr, va, te

    def score_split(seed):
        _, va, te = make_split(seed)
        worst = 0
        for part in (set(va), set(te)):
            sub = pairs_df[pairs_df.sentence_id.isin(part)]
            if not (0.14 <= len(sub) / len(pairs_df) <= 0.16):
                return 9.0
            worst = max(worst, (sub.polarity.value_counts(normalize=True) - overall).abs().max())
        return worst

    best_seed = min(range(search_seeds), key=score_split)
    tr_ids, va_ids, te_ids = make_split(best_seed)
    
    split_map = {
        **{i: 'train' for i in tr_ids},
        **{i: 'val' for i in va_ids},
        **{i: 'test' for i in te_ids}
    }
    
    sents_df = sents_df.copy()
    pairs_df = pairs_df.copy()
    sents_df['split'] = sents_df.sentence_id.map(split_map)
    pairs_df['split'] = pairs_df.sentence_id.map(split_map)
    
    return sents_df, pairs_df, best_seed


def calculate_class_weights(train_labels, classes=('negative', 'neutral', 'positive')):
    """Computes balanced class weights for the training subset."""
    weights = compute_class_weight('balanced', classes=np.array(classes), y=train_labels)
    return dict(zip(classes, weights.round(4).tolist()))
