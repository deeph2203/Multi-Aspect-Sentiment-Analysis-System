# scripts/run_step4_features.py - Run Step 4 Feature Engineering
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.features.build_features import build_all_features

if __name__ == '__main__':
    print("=== Step 4: Feature Extraction & Neural Encodings ===")
    arr, emb_matrix, vocab, pairs = build_all_features()
    print("Feature shapes:")
    for k, v in arr.items():
        print(f"  {k:<12}: {v.shape}")
    print(f"GloVe embedding matrix shape: {emb_matrix.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    print("Step 4 complete.")
