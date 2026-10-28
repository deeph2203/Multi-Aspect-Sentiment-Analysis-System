# src/models/train_asc.py - Training pipeline for Aspect Sentiment Classification (PA-BiLSTM & BiLSTM-Attn)
import copy, json, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import PROCESSED_DIR, MODELS_DIR, DEVICE, ALPHA_DECAY
from src.models.bilstm_attn import BiLSTMAttn, PositionAwareBiLSTMAttn


def train_asc_model(model_type='position_aware', seeds=(42, 100, 2026), max_epochs=35, patience=8, lr=1.2e-3, weight_decay=1e-4):
    """
    Trains the Aspect Sentiment Classification model across multiple seeds.
    Args:
        model_type (str): 'position_aware' (recommended) or 'baseline'
        seeds (tuple): list of random seeds for multi-seed stability
    """
    z = np.load(PROCESSED_DIR / 'features.npz', allow_pickle=True)
    a_dict = {k: z[k] for k in z.files}
    emb_matrix = np.load(PROCESSED_DIR / 'embedding_matrix.npy')
    vocab = json.load(open(PROCESSED_DIR / 'vocab.json'))

    y = a_dict['y']
    split = a_dict['split'].astype(str)
    tr, va, te = split == 'train', split == 'val', split == 'test'

    class_w = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y[tr])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float).to(DEVICE))

    tensor_data = dict(
        sent=torch.tensor(a_dict['sent_ids']),
        asp=torch.tensor(a_dict['asp_ids']),
        ctx=torch.tensor(a_dict['ctx_ids']),
        len=torch.tensor(a_dict['sent_len']),
        asp_start=torch.tensor(a_dict['asp_start']),
        asp_end=torch.tensor(a_dict['asp_end']),
        y=torch.tensor(y)
    )
    keys = ['sent', 'asp', 'ctx', 'len', 'asp_start', 'asp_end', 'y']

    def make_loader(mask, bs=32, shuffle=True):
        m = torch.from_numpy(mask)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*[tensor_data[k][m] for k in keys]),
            batch_size=bs,
            shuffle=shuffle
        )

    def evaluate(mod, mask):
        mod.eval()
        preds, trues = [], []
        with torch.no_grad():
            for sent, asp, ctx, ln, s_st, s_en, yb in make_loader(mask, 128, False):
                if model_type == 'position_aware':
                    logits, _ = mod(sent.to(DEVICE), asp.to(DEVICE), ln, s_st.to(DEVICE), s_en.to(DEVICE))
                else:
                    logits, _ = mod(sent.to(DEVICE), asp.to(DEVICE), ln)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(yb.numpy())
        acc = accuracy_score(trues, preds)
        macro_f1 = f1_score(trues, preds, average='macro')
        return acc, macro_f1, preds

    runs = []
    print(f"\n=== Training ASC Model: {model_type.upper()} ===")
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        if model_type == 'position_aware':
            model = PositionAwareBiLSTMAttn(
                emb_matrix.shape[0], emb_matrix.shape[1],
                alpha_decay=ALPHA_DECAY, emb_matrix=emb_matrix
            ).to(DEVICE)
        else:
            model = BiLSTMAttn(
                emb_matrix.shape[0], emb_matrix.shape[1],
                emb_matrix=emb_matrix
            ).to(DEVICE)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_f1, best_state, bad = -1, None, 0

        for ep in range(1, max_epochs + 1):
            model.train()
            for sent, asp, ctx, ln, s_st, s_en, yb in make_loader(tr, 32, True):
                if model_type == 'position_aware':
                    logits, _ = model(sent.to(DEVICE), asp.to(DEVICE), ln, s_st.to(DEVICE), s_en.to(DEVICE))
                else:
                    logits, _ = model(sent.to(DEVICE), asp.to(DEVICE), ln)
                loss = loss_fn(logits, yb.to(DEVICE))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            va_acc, va_f1, _ = evaluate(model, va)
            if va_f1 > best_f1:
                best_f1 = va_f1
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break

        model.load_state_dict(best_state)
        te_acc, te_f1, te_preds = evaluate(model, te)
        print(f"Seed {s:4d} | Val F1: {best_f1:.3f} | Test Acc: {te_acc:.3f} | Test F1: {te_f1:.3f}")
        runs.append((best_f1, te_acc, te_f1, model))

    best_val_f1, best_te_acc, best_te_f1, best_model = max(runs, key=lambda r: r[0])

    # Save artifact
    target_filename = 'asc_position_aware.pt' if model_type == 'position_aware' else 'asc_bilstm_attn.pt'
    save_path = MODELS_DIR / target_filename
    torch.save(dict(
        model='Position-Aware BiLSTM + aspect attention' if model_type == 'position_aware' else 'BiLSTM + aspect attention',
        state=best_model.state_dict(),
        vocab=vocab,
        emb_shape=emb_matrix.shape,
        val_macro_f1=float(best_val_f1),
        test_acc=float(best_te_acc),
        test_macro_f1=float(best_te_f1)
    ), save_path)
    print(f"Saved {model_type} model to: {save_path}")

    return best_model, best_val_f1, best_te_acc, best_te_f1
