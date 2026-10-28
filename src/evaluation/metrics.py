# src/evaluation/metrics.py - Evaluation metrics for sequence tagging (ATE) and sentiment (ASC)
def tags_to_spans(tags):
    """
    Converts BIO tag list to span tuples:
    ['O', 'B-ASP', 'I-ASP', 'O'] -> [(1, 3)] (start index inclusive, end index exclusive)
    """
    out, start = [], None
    for i, t in enumerate(list(tags) + ['O']):
        if t in ('B-ASP', 'O'):
            if start is not None:
                out.append((start, i))
                start = None
            if t == 'B-ASP':
                start = i
    return out


def _prf(tp_p, n_p, tp_r, n_r):
    p = tp_p / max(n_p, 1)
    r = tp_r / max(n_r, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1


def exact_prf(gold, pred):
    """
    Entity-level exact match:
    A predicted aspect is correct only if start AND end token indices match the gold aspect.
    """
    tp = n_p = n_g = 0
    for g, p in zip(gold, pred):
        gold_set = set(tags_to_spans(g))
        pred_set = set(tags_to_spans(p))
        tp += len(gold_set & pred_set)
        n_g += len(gold_set)
        n_p += len(pred_set)
    return _prf(tp, n_p, tp, n_g)


def partial_prf(gold, pred):
    """
    Partial match:
    Any token overlap between a predicted aspect and a gold aspect is considered a true positive.
    """
    overlap = lambda a, b: a[0] < b[1] and b[0] < a[1]
    tp_p = n_p = tp_r = n_r = 0
    for g, p in zip(gold, pred):
        gold_spans = tags_to_spans(g)
        pred_spans = tags_to_spans(p)
        n_p += len(pred_spans)
        n_r += len(gold_spans)
        tp_p += sum(any(overlap(x, y) for y in gold_spans) for x in pred_spans)
        tp_r += sum(any(overlap(x, y) for x in pred_spans) for y in gold_spans)
    return _prf(tp_p, n_p, tp_r, n_r)
