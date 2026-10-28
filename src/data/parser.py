# src/data/parser.py - SemEval XML dataset parser
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd


def parse_semeval(xml_path):
    """
    Parses SemEval-2014 Task 4 XML dataset file into sentences and aspect-sentiment pairs.
    Filters out 'conflict' polarity labels and NULL aspect terms.
    
    Returns:
        sents_df (pd.DataFrame): DataFrame of sentences with id, text, and span list.
        pairs_df (pd.DataFrame): DataFrame of aspect-sentiment pairs with offsets.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML dataset not found at: {xml_path}")

    sents, pairs = [], []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for s in root.iter('sentence'):
        sid = s.get('id')
        text_elem = s.find('text')
        text = text_elem.text if text_elem is not None else ""
        
        terms = s.find('aspectTerms')
        spans = []
        if terms is not None:
            for t in terms.findall('aspectTerm'):
                polarity = t.get('polarity')
                term = t.get('term')
                if polarity == 'conflict' or term == 'NULL' or not term:
                    continue
                a, b = int(t.get('from')), int(t.get('to'))
                spans.append((a, b))
                pairs.append(dict(
                    sentence_id=sid,
                    text=text,
                    term=term,
                    polarity=polarity,
                    start=a,
                    end=b
                ))
        sents.append(dict(sentence_id=sid, text=text, spans=spans))

    sents_df = pd.DataFrame(sents)
    pairs_df = pd.DataFrame(pairs)

    # Offset sanity validation
    bad_offsets = [
        1 for t, s, e, term in zip(pairs_df.text, pairs_df.start, pairs_df.end, pairs_df.term)
        if t[s:e] != term
    ]
    if bad_offsets:
        print(f"Warning: {len(bad_offsets)} rows have offset mismatches with the text.")

    return sents_df, pairs_df
