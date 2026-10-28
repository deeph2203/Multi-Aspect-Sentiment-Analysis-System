# src/evaluation/insights.py - Batch inference and business intelligence chart generation
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import BASE_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR
from src.evaluation.aspect_categories import aspect_category


def generate_business_insights(reviews=None, analyzer=None):
    """
    Runs the full pipeline on review texts and exports:
    1. insights_predictions.csv
    2. insight_top_aspects.png
    3. insight_sentiment_by_category.png
    """
    if analyzer is None:
        from src.inference.pipeline import AspectAnalyzer
        analyzer = AspectAnalyzer()

    if reviews is None:
        sents = pd.read_pickle(PROCESSED_DIR / 'sents_prep.pkl')
        reviews = sents[sents.split == 'test'].text.tolist()

    rows = []
    for r in reviews:
        for a in analyzer.analyze(r):
            rows.append(dict(
                review=r,
                aspect=a['aspect'].lower(),
                category=aspect_category(a['aspect']),
                sentiment=a['sentiment'],
                confidence=round(a['confidence'], 3)
            ))

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'insights_predictions.csv', index=False)

    # Chart 1: Most mentioned aspects
    top = df.aspect.value_counts().head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    top[::-1].plot(kind='barh', ax=ax, color='#4C72B0')
    ax.set_xlabel('mentions')
    ax.set_title('Most mentioned aspects')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'insight_top_aspects.png', dpi=200)
    plt.close()

    # Chart 2: Sentiment split per category
    named = df[df.category != 'other']
    tab = pd.crosstab(named.category, named.sentiment).reindex(columns=['negative', 'neutral', 'positive'], fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
    pct = tab.div(tab.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pct.plot(kind='barh', stacked=True, ax=ax, color={'negative': '#C44E52', 'neutral': '#B0B0B0', 'positive': '#55A868'})
    ax.invert_yaxis()
    ax.set_xlabel('% of mentions')
    ax.set_ylabel('')
    ax.set_yticklabels([f'{c} (n={n})' for c, n in zip(pct.index, tab.sum(axis=1))])
    ax.set_title('Sentiment per aspect category')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'insight_sentiment_by_category.png', dpi=200)
    plt.close()

    print(f"Generated business insight charts in: {FIGURES_DIR}")
    return df
