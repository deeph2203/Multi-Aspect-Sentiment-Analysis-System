# scripts/run_step2_preprocess.py - Run Step 2 Preprocessing & Splitting
import json, sys
from pathlib import Path
import pandas as pd
import spacy

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import PROCESSED_DIR
from src.data.preprocessor import clean_text, to_tokens, split_dataset_by_sentence, calculate_class_weights

if __name__ == '__main__':
    print("=== Step 2: Preprocessing & Splitting ===")
    nlp = spacy.load('en_core_web_sm')
    stop_words = nlp.Defaults.stop_words

    pairs = pd.read_csv(PROCESSED_DIR / 'laptop_pairs.csv', dtype={'sentence_id': str})
    sents = pd.read_pickle(PROCESSED_DIR / 'laptop_sents.pkl')

    uniq = sents.text.tolist()
    tok_map = {
        txt: to_tokens(d, stop_words)
        for txt, d in zip(uniq, nlp.pipe([clean_text(t) for t in uniq], batch_size=64))
    }
    sents['tokens'] = sents.text.map(lambda t: ' '.join(tok_map[t]))
    pairs['tokens'] = pairs.text.map(lambda t: ' '.join(tok_map[t]))

    sents, pairs, best_seed = split_dataset_by_sentence(sents, pairs)
    print(f"Optimal Split Seed: {best_seed}")

    weights = calculate_class_weights(pairs[pairs.split == 'train'].polarity)
    print("Class weights (train):", weights)

    pairs.to_csv(PROCESSED_DIR / 'pairs_prep.csv', index=False)
    sents.to_pickle(PROCESSED_DIR / 'sents_prep.pkl')
    (PROCESSED_DIR / 'class_weights.json').write_text(json.dumps(weights))

    print(f"Artifacts saved in {PROCESSED_DIR}")
