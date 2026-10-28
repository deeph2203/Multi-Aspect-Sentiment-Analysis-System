# scripts/run_step3_ate.py - Run Step 3 ATE (BIO Conversion & BiLSTM-CRF Training)
import sys
from pathlib import Path
import pandas as pd
import spacy

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import PROCESSED_DIR
from src.data.bio_tagger import to_bio
from src.models.train_ate import train_ate_model

if __name__ == '__main__':
    print("=== Step 3: Aspect Term Extraction Pipeline ===")
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])
    sents = pd.read_pickle(PROCESSED_DIR / 'sents_prep.pkl')

    data = []
    for row, doc in zip(sents.itertuples(), nlp.pipe(sents.text.tolist(), batch_size=64)):
        data.append(dict(
            sentence_id=row.sentence_id,
            split=row.split,
            text=row.text,
            spans=row.spans,
            tokens=[t.text for t in doc],
            pos=[t.pos_ for t in doc],
            tags=to_bio(doc, row.spans)
        ))
    bio_df = pd.DataFrame(data)
    bio_df.to_pickle(PROCESSED_DIR / 'bio_data.pkl')
    print(f"BIO tags prepared and saved to {PROCESSED_DIR / 'bio_data.pkl'}")

    # Train BiLSTM-CRF
    train_ate_model()
