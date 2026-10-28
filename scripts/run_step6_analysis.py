# scripts/run_step6_analysis.py - Run Step 6 Error Analysis & Per-Category Breakdown
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluation.error_analysis import run_error_analysis

if __name__ == '__main__':
    print("=== Step 6: Error Analysis ===")
    run_error_analysis()
