# src/models/train_ate.py - Training pipeline for Aspect Term Extraction (BiLSTM-CRF)
from collections import Counter
import copy, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from seqeval.metrics import classification_report

from src.config import (
    PROCESSED_DIR, TABLES_DIR, MODELS_DIR, GLOVE_FILE,
    SEED, W_DIM, P_DIM, HID, DROP, TAG2ID, ID2TAG, DEVICE
)
from src.evaluation.metrics import exact_prf, partial_prf, tags_to_spans
from src.data.bio_tagger import rule_tags
from src.models.bilstm_crf import BiLSTMCRF


def train_ate_model(max_epochs=40, patience=8, lr=2e-3, word_drop=0.1, batch_size=32):
    """Trains the BiLSTM-CRF sequence tagger on bio_data.pkl with early stopping."""
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    
    bio = pd.read_pickle(PROCESSED_DIR / 'bio_data.pkl')
    train_df = bio[bio.split == 'train']

    cnt = Counter(w.lower() for toks in train_df.tokens for w in toks)
    word2id = {'<pad>': 0, '<unk>': 1}
    for w in cnt:
        word2id[w] = len(word2id)
    pos2id = {'<pad>': 0, **{p: i + 1 for i, p in enumerate(sorted({p for ps in bio.pos for p in ps}))}}

    def encode(df):
        return [
            ([word2id.get(w.lower(), 1) for w in toks], [pos2id[p] for p in pos], [TAG2ID[t] for t in tags])
            for toks, pos, tags in zip(df.tokens, df.pos, df.tags)
        ]

    def make_batches(data, bs, shuffle):
        idx = list(range(len(data)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [data[j] for j in idx[i:i + bs]]
            max_len = max(len(c[0]) for c in chunk)
            pad = lambda s: s + [0] * (max_len - len(s))
            w = torch.tensor([pad(c[0]) for c in chunk])
            p = torch.tensor([pad(c[1]) for c in chunk])
            t = torch.tensor([pad(c[2]) for c in chunk])
            m = torch.tensor([[1] * len(c[0]) + [0] * (max_len - len(c[0])) for c in chunk], dtype=torch.bool)
            yield w.to(DEVICE), p.to(DEVICE), t.to(DEVICE), m.to(DEVICE)

    data = {sp: encode(bio[bio.split == sp]) for sp in ('train', 'val', 'test')}
    gold = {sp: bio[bio.split == sp].tags.tolist() for sp in ('val', 'test')}

    # Optional GloVe initialization
    emb = None
    if GLOVE_FILE.exists():
        emb = np.random.normal(0, 0.1, (len(word2id), W_DIM)).astype('float32')
        emb[0] = 0
        hit = 0
        with open(GLOVE_FILE, encoding='utf8') as f:
            for line in f:
                w, *vec = line.rstrip().split(' ')
                if w in word2id and len(vec) == W_DIM:
                    emb[word2id[w]] = np.asarray(vec, dtype='float32')
                    hit += 1
        print(f"GloVe loaded: {hit}/{len(word2id)} words initialized.")

    model = BiLSTMCRF(len(word2id), len(pos2id), W_DIM, P_DIM, HID, DROP, emb).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def predict_tags(mod, d):
        mod.eval()
        out = []
        with torch.no_grad():
            for w, p, t, m in make_batches(d, 64, False):
                out += [[ID2TAG[i] for i in seq] for seq in mod.predict(w, p, m)]
        return out

    best_f1, best_state, bad = -1, None, 0
    print("\n=== Training BiLSTM-CRF for Aspect Extraction ===")
    for ep in range(1, max_epochs + 1):
        t0, losses = time.time(), []
        model.train()
        for w, p, t, m in make_batches(data['train'], batch_size, True):
            w = torch.where((torch.rand(w.shape, device=w.device) < word_drop) & (w != 0), torch.ones_like(w), w)
            loss = model.loss(w, p, t, m)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        val_f1 = exact_prf(gold['val'], predict_tags(model, data['val']))[2]
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
            flag = ' <- best'
        else:
            bad += 1
            flag = ''

        if ep % 5 == 0 or bad == 0:
            print(f"Epoch {ep:2d} | Loss: {np.mean(losses):.3f} | Val Exact F1: {val_f1:.3f} | {time.time()-t0:.1f}s{flag}")
        if bad >= patience:
            print(f"Early stopped at epoch {ep}")
            break

    model.load_state_dict(best_state)

    # Evaluation
    rows = []
    test_pred = None
    for sp in ('val', 'test'):
        d = bio[bio.split == sp]
        systems = {
            'Rule baseline (nouns)': [rule_tags(p) for p in d.pos],
            'BiLSTM-CRF': predict_tags(model, data[sp])
        }
        for name, pred in systems.items():
            (pe, re_, fe), (pp, rp, fp) = exact_prf(gold[sp], pred), partial_prf(gold[sp], pred)
            rows.append(dict(model=name, split=sp, exact_P=pe, exact_R=re_, exact_F1=fe, partial_P=pp, partial_R=rp, partial_F1=fp))
        if sp == 'test':
            test_pred = systems['BiLSTM-CRF']

    res = pd.DataFrame(rows).round(3)
    res.to_csv(TABLES_DIR / 'ate_results.csv', index=False)

    # Save model checkpoint
    torch.save(dict(
        state=model.state_dict(),
        word2id=word2id,
        pos2id=pos2id,
        cfg=dict(W_DIM=W_DIM, P_DIM=P_DIM, HID=HID, DROP=DROP)
    ), MODELS_DIR / 'ate_bilstm_crf.pt')

    print(f"Saved checkpoint to: {MODELS_DIR / 'ate_bilstm_crf.pt'}")
    return model, res
