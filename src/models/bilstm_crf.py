# src/models/bilstm_crf.py - BiLSTM-CRF sequence tagger for Aspect Term Extraction (ATE)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class BiLSTMCRF(nn.Module):
    """
    Bidirectional LSTM with Conditional Random Field (CRF) layer for ATE.
    Combines word embeddings with POS tag embeddings to accurately mark BIO spans.
    """
    def __init__(self, n_words, n_pos, w_dim=100, p_dim=25, hid=128, drop=0.5, emb=None):
        super().__init__()
        self.wemb = nn.Embedding(n_words, w_dim, padding_idx=0)
        if emb is not None:
            self.wemb.weight.data.copy_(torch.tensor(emb))
        self.pemb = nn.Embedding(n_pos, p_dim, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.lstm = nn.LSTM(w_dim + p_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hid, 3)
        self.crf = CRF(3, batch_first=True)

    def _emit(self, w, p, mask):
        x = self.drop(torch.cat([self.wemb(w), self.pemb(p)], -1))
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=w.size(1))
        return self.fc(self.drop(h))

    def loss(self, w, p, tags, mask):
        """Computes negative log-likelihood loss from the CRF layer."""
        emissions = self._emit(w, p, mask)
        return -self.crf(emissions, tags, mask=mask, reduction='mean')

    def predict(self, w, p, mask):
        """Decodes the optimal Viterbi sequence of tag indices."""
        emissions = self._emit(w, p, mask)
        return self.crf.decode(emissions, mask=mask)
