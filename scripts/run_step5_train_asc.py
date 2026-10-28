# scripts/run_step5_train_asc.py - Run Step 5 ASC Training (Position-Aware & Baseline)
import argparse, sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.train_asc import train_asc_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Aspect Sentiment Classification Model")
    parser.add_argument(
        '--model',
        choices=['position_aware', 'baseline', 'both'],
        default='position_aware',
        help="Model architecture to train (default: position_aware)"
    )
    args = parser.parse_args()

    if args.model in ('position_aware', 'both'):
        print("\nTraining Position-Aware BiLSTM...")
        train_asc_model(model_type='position_aware')

    if args.model in ('baseline', 'both'):
        print("\nTraining Baseline BiLSTM-Attn...")
        train_asc_model(model_type='baseline')

    print("Step 5 complete.")
