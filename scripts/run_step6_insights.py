# scripts/run_step6_insights.py - Run Step 6 Business Insights & Charts
import argparse, sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluation.insights import generate_business_insights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Business Insights and Charts")
    parser.add_argument('--file', help="Optional text file with one review per line")
    args = parser.parse_args()

    reviews = None
    if args.file:
        with open(args.file, encoding='utf8') as f:
            reviews = [l.strip() for l in f if l.strip()]

    print("=== Step 6: Generating Business Insights & Charts ===")
    generate_business_insights(reviews=reviews)
