# src/evaluation/error_analysis.py - Per-category accuracy and error diagnostics
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from src.config import TABLES_DIR, FIGURES_DIR
from src.evaluation.aspect_categories import aspect_category

NEG_WORDS = {'not', 'no', 'never', 'nothing', 'without', 'nor', 'cannot', 'hardly', 'barely'}
CONTRAST_WORDS = {'but', 'however', 'although', 'though', 'while', 'yet', 'except', 'despite', 'whereas'}


def flags(row, col):
    """Generates diagnostic tags for error analysis."""
    toks = re.findall(r"[a-z']+", row.text.lower())
    f = []
    if any(t in NEG_WORDS or t.endswith("n't") for t in toks):
        f.append('negation')
    if any(t in CONTRAST_WORDS for t in toks):
        f.append('contrast')
    if row.mixed:
        f.append('mixed-aspects')
    if 'neutral' in (row.polarity, row[col]):
        f.append('neutral-boundary')
    if len(toks) > 30:
        f.append('long-sentence')
    return f


def run_error_analysis():
    """Executes per-category accuracy breakdown, mixed-sentence analysis, and error sampling."""
    pred_path = TABLES_DIR / 'test_predictions.csv'
    comp_path = TABLES_DIR / 'model_comparison.csv'
    if not pred_path.exists() or not comp_path.exists():
        print("Predictions or comparison table missing. Run model evaluation first.")
        return

    pred = pd.read_csv(pred_path, dtype={'sentence_id': str})
    comp = pd.read_csv(comp_path)

    trained = [m for m in comp.model if 'Rule' not in m and 'pred_' + m in pred.columns]
    best_model = comp[comp.model.isin(trained)].sort_values('val_macro_f1', ascending=False).model.iloc[0]
    col = 'pred_' + best_model

    pred['category'] = pred.term.map(aspect_category)
    pred['ok'] = pred[col] == pred.polarity

    # Per-category results
    rows = []
    for cat, g in pred.groupby('category'):
        rows.append(dict(
            category=cat,
            n=len(g),
            accuracy=g.ok.mean(),
            macro_f1=f1_score(g.polarity, g[col], average='macro') if g.polarity.nunique() > 1 else np.nan
        ))
    cat_tab = pd.DataFrame(rows).sort_values('n', ascending=False).round(3)
    cat_tab.to_csv(TABLES_DIR / 'per_category_results.csv', index=False)

    # Plot
    plot = cat_tab[cat_tab.category != 'other']
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(plot.category, plot.accuracy, color='#4C72B0')
    for b, n in zip(bars, plot.n):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f'n={n}', va='center', fontsize=8)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel('accuracy')
    ax.set_title(f'{best_model}: accuracy per aspect category (test)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'per_category_accuracy.png', dpi=200)
    plt.close()

    # Mixed-sentence diagnostics
    g = pred.groupby('sentence_id').polarity.agg(['nunique', 'size'])
    pred['n_aspects'] = pred.sentence_id.map(g['size'])
    pred['mixed'] = pred.sentence_id.map(g['nunique'] > 1)

    # Error sampling
    sample = pred[~pred.ok].sample(n=min(30, (~pred.ok).sum()), random_state=42)
    sample['flag_list'] = sample.apply(lambda r: flags(r, col), axis=1)
    out = sample[['sentence_id', 'text', 'term', 'polarity', col, 'flag_list']].rename(
        columns={col: 'predicted', 'polarity': 'gold', 'flag_list': 'auto_flags'}
    )
    out['auto_flags'] = out['auto_flags'].map(lambda f: ', '.join(f))
    out['your_cause'] = ''
    out.to_csv(TABLES_DIR / 'error_analysis_sample.csv', index=False)

    print(f"Error analysis completed. Outputs saved to {TABLES_DIR} and {FIGURES_DIR}.")
    return cat_tab
