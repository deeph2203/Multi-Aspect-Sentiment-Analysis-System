# src/inference/pipeline.py - Production inference pipeline for Aspect-Based Sentiment Analysis
from pathlib import Path
import spacy
import torch

from src.config import MODELS_DIR, LABELS, ID2TAG, MAX_LEN, MAX_ASP
from src.evaluation.metrics import tags_to_spans
from src.models.bilstm_crf import BiLSTMCRF
from src.models.bilstm_attn import BiLSTMAttn, PositionAwareBiLSTMAttn


class AspectAnalyzer:
    """
    End-to-End Aspect-Based Sentiment Analysis Pipeline:
    1. Extracts aspect spans using a BiLSTM-CRF sequence tagger.
    2. Classifies each aspect's sentiment using an aspect-attentive BiLSTM model.
    Supports both the Improved Position-Aware BiLSTM (recommended) and the Baseline BiLSTM-Attn.
    """
    def __init__(self, model_mode='auto'):
        """
        model_mode: 'auto' (loads position_aware if available, else original),
                    'position_aware' (forces improved PA-BiLSTM),
                    'original' (forces baseline BiLSTM-Attn)
        """
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])
        
        # Load ATE (Aspect Term Extraction) model
        ate_path = MODELS_DIR / 'ate_bilstm_crf.pt'
        ate = torch.load(ate_path, map_location='cpu', weights_only=False)
        c = ate['cfg']
        self.ate_w2i, self.ate_p2i = ate['word2id'], ate['pos2id']
        self.ate = BiLSTMCRF(len(self.ate_w2i), len(self.ate_p2i), c['W_DIM'], c['P_DIM'], c['HID'], c['DROP'])
        self.ate.load_state_dict(ate['state'])
        self.ate.eval()

        # Load ASC (Aspect Sentiment Classification) model
        pa_path = MODELS_DIR / 'asc_position_aware.pt'
        orig_path = MODELS_DIR / 'asc_bilstm_attn.pt'

        if (model_mode in ('auto', 'position_aware')) and pa_path.exists():
            asc = torch.load(pa_path, map_location='cpu', weights_only=False)
            self.vocab = asc['vocab']
            n_vocab, d = asc['emb_shape']
            self.asc = PositionAwareBiLSTMAttn(n_vocab, d)
            self.asc.load_state_dict(asc['state'])
            self.asc.eval()
            self.is_position_aware = True
            self.model_name = 'Position-Aware BiLSTM + aspect attention'
        else:
            asc = torch.load(orig_path, map_location='cpu', weights_only=False)
            self.vocab = asc['vocab']
            n_vocab, d = asc['emb_shape']
            self.asc = BiLSTMAttn(n_vocab, d)
            self.asc.load_state_dict(asc['state'])
            self.asc.eval()
            self.is_position_aware = False
            self.model_name = 'BiLSTM + aspect attention (baseline)'

    @torch.no_grad()
    def extract(self, doc):
        """Extracts aspect token spans from a processed spaCy document."""
        w = torch.tensor([[self.ate_w2i.get(t.text.lower(), 1) for t in doc]])
        p = torch.tensor([[self.ate_p2i.get(t.pos_, 0) for t in doc]])
        mask = torch.ones_like(w, dtype=torch.bool)
        tags = [ID2TAG[i] for i in self.ate.predict(w, p, mask)[0]]
        return tags_to_spans(tags)

    @torch.no_grad()
    def classify(self, doc, a, b):
        """Classifies sentiment for a single aspect span [a, b)."""
        toks = [t.lower_ for t in doc]
        n = len(toks)
        idx = list(range(a, b))
        s0 = max(0, min(idx[0] - 10, n - MAX_LEN)) if n > MAX_LEN else 0
        sent = toks[s0:s0 + MAX_LEN]
        ids = [self.vocab.get(t, 1) for t in sent]
        asp = [self.vocab.get(toks[i], 1) for i in idx][:MAX_ASP]
        asp += [0] * (MAX_ASP - len(asp))

        rel_start = max(0, idx[0] - s0)
        rel_end = min(idx[-1] - s0 + 1, len(sent))

        if self.is_position_aware:
            logits, att = self.asc(
                torch.tensor([ids]),
                torch.tensor([asp]),
                torch.tensor([len(ids)]),
                torch.tensor([rel_start]),
                torch.tensor([rel_end])
            )
        else:
            logits, att = self.asc(
                torch.tensor([ids]),
                torch.tensor([asp]),
                torch.tensor([len(ids)])
            )

        probs = torch.softmax(logits, 1)[0]
        k = int(probs.argmax())
        order = att[0].argsort(descending=True).tolist()
        top = [sent[i] for i in order if sent[i].isalpha() and (s0 + i) not in idx][:3]

        return dict(
            aspect=doc[a:b].text,
            start_char=doc[a].idx,
            end_char=doc[b - 1].idx + len(doc[b - 1]),
            sentiment=LABELS[k],
            confidence=float(probs[k]),
            probs={l: float(p) for l, p in zip(LABELS, probs)},
            key_words=top
        )

    def analyze(self, text):
        """Analyzes a free-text product review and returns detected aspects with sentiments."""
        text = (text or '').strip()
        if not text:
            return []
        doc = self.nlp(text)
        return [self.classify(doc, a, b) for a, b in self.extract(doc)]
