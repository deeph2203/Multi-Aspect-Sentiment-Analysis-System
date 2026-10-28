# scripts/run_step1_data.py - Run Step 1 Data Parsing & EDA
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import RAW_DIR, PROCESSED_DIR, FIGURES_DIR
from src.data.parser import parse_semeval

if __name__ == '__main__':
    print("=== Step 1: Parsing SemEval Dataset ===")
    xml_file = RAW_DIR / 'laptops_train.xml'
    sents, pairs = parse_semeval(xml_file)

    pairs.to_csv(PROCESSED_DIR / 'laptop_pairs.csv', index=False)
    sents.to_pickle(PROCESSED_DIR / 'laptop_sents.pkl')

    print(f"Total Sentences: {len(sents)} | Total Pairs: {len(pairs)}")
    print("\nSentiment distribution:\n", pairs.polarity.value_counts())

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    pairs.polarity.value_counts().plot(kind='bar', ax=ax[0], title='Sentiment distribution')
    pairs.term.str.lower().value_counts().head(15).plot(kind='barh', ax=ax[1], title='Top 15 aspects')
    sents.text.str.split().str.len().plot(kind='hist', bins=30, ax=ax[2], title='Words per sentence')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eda_overview.png', dpi=200)
    plt.close()

    print(f"Artifacts saved in {PROCESSED_DIR} and {FIGURES_DIR}")
