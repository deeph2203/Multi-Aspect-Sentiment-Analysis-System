# src/models/bilstm_attn.py - BiLSTM with Aspect Attention architectures (Baseline and Position-Aware)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMAttn(nn.Module):
    """
    Baseline BiLSTM with Aspect Attention (from Step 5).
    Concatenates the aspect vector with each word and computes global attention weights.
    """
    def __init__(self, n_vocab, d=100, hid=128, drop=0.5, emb_matrix=None):
        super().__init__()
        if emb_matrix is not None:
            self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=False, padding_idx=0)
        else:
            self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.lstm = nn.LSTM(d * 2, hid, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hid + d, 1)
        self.drop, self.fc = nn.Dropout(drop), nn.Linear(2 * hid, 3)

    def forward(self, sent, asp, length, asp_start=None, asp_end=None):
        x, mask = self.emb(sent), sent != 0
        m = (asp != 0).float().unsqueeze(-1)
        a = ((self.emb(asp) * m).sum(1) / m.sum(1).clamp(min=1)).unsqueeze(1).expand(-1, x.size(1), -1)
        h, _ = self.lstm(pack_padded_sequence(self.drop(torch.cat([x, a], -1)), length.cpu(), batch_first=True, enforce_sorted=False))
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=sent.size(1))
        score = self.att(torch.cat([h, a], -1)).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(score, dim=1).unsqueeze(-1)
        return self.fc(self.drop((w * h).sum(1))), w.squeeze(-1)


class PositionAwareBiLSTMAttn(nn.Module):
    """
    Position-Aware Aspect-Based Sentiment BiLSTM with Attention (PA-BiLSTM).
    Enhancements over baseline:
    1. Frozen GloVe embeddings prevent aspect words from memorizing class labels.
    2. Relative position embeddings map token distances relative to the aspect span.
    3. Distance-decay penalty directly prevents cross-aspect attention leakage in mixed sentences.
    """
    def __init__(self, n_vocab, d=100, hid=128, drop=0.5, pos_dim=25, alpha_decay=0.18, emb_matrix=None):
        super().__init__()
        self.hid = hid
        self.alpha_decay = alpha_decay
        if emb_matrix is not None:
            self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=True, padding_idx=0)
        else:
            self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
            
        self.pos_emb = nn.Embedding(121, pos_dim, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.lstm = nn.LSTM(d * 2 + pos_dim, hid, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(
            nn.Linear(2 * hid + d + pos_dim, hid),
            nn.Tanh(),
            nn.Linear(hid, 1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(2 * hid, hid),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid, 3)
        )

    def forward(self, sent, asp, length, asp_start=None, asp_end=None):
        batch_size, seq_len = sent.size()
        x, mask = self.emb(sent), sent != 0
        
        # Mean aspect representation
        m = (asp != 0).float().unsqueeze(-1)
        a = ((self.emb(asp) * m).sum(1) / m.sum(1).clamp(min=1)).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Handle aspect coordinates
        if asp_start is None or asp_end is None:
            asp_start = torch.zeros(batch_size, device=sent.device)
            asp_end = torch.ones(batch_size, device=sent.device)
            
        s_mid = ((asp_start.float() + asp_end.float()) / 2.0).unsqueeze(1)
        pos_idx = torch.arange(seq_len, device=sent.device).unsqueeze(0).expand(batch_size, -1).float()
        
        dist = torch.abs(pos_idx - s_mid)
        rel_pos = torch.clamp((pos_idx - s_mid).long() + 60, 0, 120)
        p_emb = self.pos_emb(rel_pos)
        
        # Distance-decay attention penalty: suppresses distant clause tokens
        dist_penalty = -self.alpha_decay * dist
        
        # Bidirectional LSTM encoding
        lstm_in = torch.cat([x, a, p_emb], dim=-1)
        packed = pack_padded_sequence(self.drop(lstm_in), length.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=seq_len)
        
        # Score attention
        att_features = torch.cat([h, a, p_emb], dim=-1)
        score = self.att(att_features).squeeze(-1) + dist_penalty
        score = score.masked_fill(~mask, -1e9)
        w = torch.softmax(score, dim=1).unsqueeze(-1)
        
        # Aggregate attentive context vector
        ctx_vec = (w * h).sum(1)
        return self.fc(ctx_vec), w.squeeze(-1)
