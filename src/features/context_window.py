# src/features/context_window.py - Aspect context windows and dependency-based opinion words
from src.data.preprocessor import KEEP_WORDS


def aspect_idx(doc, start, end):
    """Returns the token indices in doc that overlap with character range [start, end]."""
    return [t.i for t in doc if t.idx < end and t.idx + len(t) > start]


def window_tokens(doc, idx, k=5, stop_words=None):
    """
    Extracts tokens within a +-k token window around the aspect indices `idx`.
    Preserves negation words and aspect tokens.
    """
    if not idx:
        return []
    lo, hi = max(idx[0] - k, 0), min(idx[-1] + k + 1, len(doc))
    return [
        'not' if t.text == "n't" else t.lemma_.lower()
        for t in doc[lo:hi]
        if (t.is_alpha or t.text == "n't") and (t.i in idx or (stop_words is None or t.lower_ not in stop_words or t.lower_ in KEEP_WORDS))
    ]


def opinion_words(doc, idx):
    """
    Extracts dependency-linked opinion candidates for the aspect tokens in `idx`.
    Traverses children (adjectives, adverbs, negation) and head verbs/predicates.
    """
    cand = []
    for i in idx:
        if i >= len(doc):
            continue
        a = doc[i]
        # Direct modifiers (e.g., 'great' in 'great battery')
        cand += [c for c in a.children if c.pos_ in ('ADJ', 'ADV')]
        # Head predicate (e.g., 'scratches' in 'screen scratches')
        if a.head.i not in idx and a.head.pos_ in ('ADJ', 'VERB'):
            cand.append(a.head)
        # Head complement/negation (e.g., 'is great', 'is not good')
        cand += [
            c for c in a.head.children
            if c.dep_ in ('acomp', 'attr', 'advmod', 'neg') and c.i not in idx
        ]

    out = []
    for t in cand:
        w = 'not' if t.text == "n't" else t.lemma_.lower()
        if w.isalpha() and w not in out:
            out.append(w)
    return out
