# src/data/bio_tagger.py - BIO sequence tagging conversion and rule baseline
def to_bio(doc, spans):
    """
    Converts character span offsets into token-level BIO tags.
    Tags: 'O' (outside), 'B-ASP' (beginning), 'I-ASP' (inside).
    """
    tags = ['O'] * len(doc)
    for s, e in spans:
        # Find tokens that overlap with the character span [s, e]
        inside = [t.i for t in doc if t.idx < e and t.idx + len(t) > s]
        for k, i in enumerate(inside):
            tags[i] = 'B-ASP' if k == 0 else 'I-ASP'
    return tags


def rule_tags(pos_tags):
    """
    Rule-based aspect extraction baseline:
    Treats every contiguous run of NOUN or PROPN tokens as an aspect term.
    """
    tags, prev = [], False
    for p in pos_tags:
        is_n = p in ('NOUN', 'PROPN')
        tags.append(('I-ASP' if prev else 'B-ASP') if is_n else 'O')
        prev = is_n
    return tags
