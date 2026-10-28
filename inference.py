# inference.py - Backward-compatible root export for the ABSA pipeline
"""
Root entrypoint for inference.
All core components are modularized under `src/`.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.inference.pipeline import AspectAnalyzer
from src.models.bilstm_crf import BiLSTMCRF
from src.models.bilstm_attn import BiLSTMAttn, PositionAwareBiLSTMAttn
from src.config import LABELS, ID2TAG, TAG2ID, MAX_LEN, MAX_ASP, K, MODELS_DIR

__all__ = [
    'AspectAnalyzer',
    'BiLSTMCRF',
    'BiLSTMAttn',
    'PositionAwareBiLSTMAttn',
    'LABELS',
    'ID2TAG',
    'TAG2ID',
    'MAX_LEN',
    'MAX_ASP',
    'K',
    'MODELS_DIR'
]

if __name__ == '__main__':
    analyzer = AspectAnalyzer()
    print(f"Loaded: {analyzer.model_name}")
    sample = "The screen is gorgeous but the battery dies in two hours."
    print(f"\nAnalyzing: \"{sample}\"")
    for r in analyzer.analyze(sample):
        print(f"  -> {r['aspect']}: {r['sentiment']} ({r['confidence']:.2f})")
