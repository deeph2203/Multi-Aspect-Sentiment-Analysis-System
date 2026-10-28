# app/app.py - Streamlit demo.  Run from the project folder:   streamlit run app/app.py
import sys, html
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.inference.pipeline import AspectAnalyzer          # noqa: E402

COLORS = {'positive': '#55A868', 'negative': '#C44E52', 'neutral': '#8C8C8C'}
EXAMPLES = {
    'Pick an example...': '',
    'Simple, one opinion': 'The battery life is excellent.',
    'Two opinions': 'The screen is gorgeous but the battery dies in two hours.',
    'Service / warranty': 'Customer service was terrible and the warranty is useless.',
    'Hard case: contrasts in one sentence': 'The battery life is great but the screen scratches easily and the price is too high.',
}

st.set_page_config(page_title='Product Review Analyzer', page_icon='🔍', layout='wide')

@st.cache_resource(show_spinner='Loading the models...')
def load_analyzer(mode='position_aware'):
    return AspectAnalyzer(model_mode=mode)

def model_accuracy(mode='position_aware'):
    f = ROOT / 'outputs' / 'tables' / 'model_comparison.csv'
    if not f.exists():
        return None
    target_name = 'Position-Aware BiLSTM' if mode == 'position_aware' else 'BiLSTM + aspect attention'
    row = pd.read_csv(f).query(f"model == '{target_name}'")
    if row.empty:
        row = pd.read_csv(f).query("model == 'BiLSTM + aspect attention'")
    return None if row.empty else float(row.test_acc.iloc[0])

def pick_example():
    st.session_state.review = EXAMPLES[st.session_state.example]

st.title('🔍 Product Review Analyzer')
st.caption('Aspect-based sentiment: it finds the product features in a review and judges each one separately.')

with st.sidebar:
    st.header('Model Selection')
    model_choice = st.selectbox(
        'Sentiment Model Architecture',
        ['Position-Aware BiLSTM (Recommended)', 'Original BiLSTM-Attn (Baseline)'],
        help='Position-Aware model uses distance-decay attention to correctly handle mixed-sentiment sentences.'
    )
    selected_mode = 'position_aware' if 'Position-Aware' in model_choice else 'original'
    
    st.header('How it works')
    st.markdown('1. **Aspect extraction**: a BiLSTM-CRF tagger finds the aspect words.\n'
                '2. **Sentiment**: a BiLSTM with aspect attention classifies each aspect as positive, neutral or negative.')
    acc = model_accuracy(selected_mode)
    st.header('Know its limits')
    st.markdown('Trained on about 2,300 laptop-review aspects (SemEval-2014), so it works best on **laptop reviews**.'
                + (f' Test accuracy of current model: **{acc:.0%}**.' if acc else '')
                + ' The Position-Aware model incorporates distance penalties to prevent cross-aspect sentiment leakage.')

st.selectbox('Examples', list(EXAMPLES), key='example', on_change=pick_example)
review = st.text_area('Paste a laptop review (English)', key='review', height=130,
                      placeholder='e.g. The screen is gorgeous but the keyboard feels cheap.')

if st.button('Analyze', type='primary') and review.strip():
    results = load_analyzer(selected_mode).analyze(review)
    if not results:
        st.info('No product aspects were found in this review.')
    else:
        # the review with each aspect highlighted in its sentiment colour
        out, pos = '', 0
        for r in sorted(results, key=lambda r: r['start_char']):
            if r['start_char'] < pos:
                continue
            out += html.escape(review[pos:r['start_char']])
            out += (f"<span style='background:{COLORS[r['sentiment']]}33;border-bottom:3px solid {COLORS[r['sentiment']]};"
                    f"padding:1px 3px;border-radius:3px'>{html.escape(review[r['start_char']:r['end_char']])}</span>")
            pos = r['end_char']
        st.markdown(f"<div style='font-size:1.1rem;line-height:1.9'>{out + html.escape(review[pos:])}</div>", unsafe_allow_html=True)

        table = pd.DataFrame([dict(Aspect=r['aspect'], Sentiment=r['sentiment'], Confidence=f"{r['confidence']:.0%}",
                                   **{'Words it looked at': ', '.join(r['key_words'])}) for r in results])
        left, right = st.columns([3, 2])
        left.dataframe(table, hide_index=True, width='stretch')
        counts = pd.Series([r['sentiment'] for r in results]).value_counts().reindex(['positive', 'neutral', 'negative'], fill_value=0)
        right.bar_chart(counts)
        with st.expander('Probabilities for each aspect'):
            st.dataframe(pd.DataFrame([dict(Aspect=r['aspect'], **{k: f'{v:.0%}' for k, v in r['probs'].items()}) for r in results]),
                         hide_index=True, width='stretch')
