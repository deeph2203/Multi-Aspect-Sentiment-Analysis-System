# backend/server.py - FastAPI backend for Aspect-Based Sentiment Analysis
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, TABLES_DIR, MODELS_DIR
from src.inference.pipeline import AspectAnalyzer
from src.evaluation.aspect_categories import aspect_category
from tests.test_pipeline import TEST_CASES

app = FastAPI(
    title="Multi-Aspect Sentiment Analysis System API",
    description="Production API serving two-stage ABSA: BiLSTM-CRF (ATE) and Position-Aware BiLSTM-Attn (ASC)",
    version="2.0.0"
)

# Enable CORS for frontend Vite dev server (and any localhost port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers lazily or at startup
analyzers: Dict[str, AspectAnalyzer] = {}

def get_analyzer(model_mode: str = 'position_aware') -> AspectAnalyzer:
    if model_mode not in analyzers:
        analyzers[model_mode] = AspectAnalyzer(model_mode=model_mode)
    return analyzers[model_mode]


class AnalyzeRequest(BaseModel):
    text: str
    model_type: Optional[str] = "position_aware"  # 'position_aware' or 'original'


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "models_available": {
            "position_aware": (MODELS_DIR / "asc_position_aware.pt").exists(),
            "baseline": (MODELS_DIR / "asc_bilstm_attn.pt").exists(),
            "ate_crf": (MODELS_DIR / "ate_bilstm_crf.pt").exists()
        },
        "device": "cpu"
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    model_mode = "position_aware" if req.model_type == "position_aware" else "original"
    analyzer = get_analyzer(model_mode)
    
    t0 = time.perf_counter()
    doc = analyzer.nlp(text)
    spans = analyzer.extract(doc)
    
    results = []
    for a, b in spans:
        clf = analyzer.classify(doc, a, b)
        term = clf["aspect"]
        cat = aspect_category(term)
        
        # Calculate relative attention tokens
        toks = [t.lower_ for t in doc]
        n = len(toks)
        idx = list(range(a, b))
        s0 = max(0, min(idx[0] - 10, n - 40)) if n > 40 else 0
        window_tokens = [t.text for t in doc[s0:s0 + 40]]
        
        results.append({
            "aspect": term,
            "category": cat,
            "start_char": clf["start_char"],
            "end_char": clf["end_char"],
            "sentiment": clf["sentiment"],
            "confidence": round(clf["confidence"], 4),
            "probabilities": {k: round(v, 4) for k, v in clf["probs"].items()},
            "key_words": clf["key_words"],
            "window_tokens": window_tokens,
            "window_offset": s0
        })
    
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    pos_count = sum(1 for r in results if r["sentiment"] == "positive")
    neg_count = sum(1 for r in results if r["sentiment"] == "negative")
    neu_count = sum(1 for r in results if r["sentiment"] == "neutral")
    
    if len(results) == 0:
        overall = "no_aspects_detected"
    elif pos_count > 0 and neg_count > 0:
        overall = "mixed"
    elif pos_count > 0 and neg_count == 0 and neu_count == 0:
        overall = "positive"
    elif neg_count > 0 and pos_count == 0 and neu_count == 0:
        overall = "negative"
    else:
        overall = "predominantly_" + ("positive" if pos_count >= neg_count else "negative")
        
    return {
        "text": text,
        "model_type": model_mode,
        "model_name": analyzer.model_name,
        "elapsed_ms": elapsed_ms,
        "aspects": results,
        "summary": {
            "total_aspects": len(results),
            "positive": pos_count,
            "negative": neg_count,
            "neutral": neu_count,
            "overall_sentiment": overall
        }
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    # Read model comparison table
    comp_file = TABLES_DIR / "model_comparison.csv"
    models_data = []
    if comp_file.exists():
        import csv
        with open(comp_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            models_data = list(reader)
            
    # Read per category results
    cat_file = TABLES_DIR / "per_category_results.csv"
    cat_data = []
    if cat_file.exists():
        import csv
        with open(cat_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cat_data = list(reader)
            
    # Read ATE results
    ate_file = TABLES_DIR / "ate_results.csv"
    ate_data = []
    if ate_file.exists():
        import csv
        with open(ate_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ate_data = list(reader)

    return {
        "models_comparison": models_data,
        "per_category": cat_data,
        "ate_metrics": ate_data,
        "test_suite_cases": TEST_CASES,
        "key_highlights": {
            "ate_exact_f1": "74.2%",
            "ate_partial_f1": "83.2%",
            "ate_rule_baseline_exact": "31.6%",
            "asc_accuracy": "64.1%",
            "asc_macro_f1": "0.617",
            "baseline_accuracy": "60.4%",
            "baseline_macro_f1": "0.565",
            "mixed_sentiment_pass_rate": "60.0% (vs 0% on baseline)"
        }
    }


@app.get("/api/pipeline")
def get_pipeline_info():
    return {
        "dataset": {
            "name": "SemEval-2014 Task 4 (Subtask Laptop Reviews)",
            "total_sentences": 3045,
            "aspect_pairs": 2313,
            "splits": {
                "train": "1,622 pairs (70%)",
                "val": "348 pairs (15%)",
                "test": "343 pairs (15%)"
            },
            "polarities": {
                "positive": "56.4%",
                "negative": "29.8%",
                "neutral": "13.8%"
            }
        },
        "stage1_ate": {
            "name": "Aspect Term Extraction (ATE)",
            "architecture": "BiLSTM-CRF with Linear-Chain Viterbi Decoding",
            "input_features": "spaCy Tokens (100d word embedding) + Universal POS Tags (30d embedding)",
            "hidden_dim": 128,
            "dropout": 0.3,
            "tagset": ["O", "B-ASP", "I-ASP"],
            "loss": "Negative Log-Likelihood over CRF forward-backward transitions",
            "test_f1_exact": 0.742,
            "test_f1_partial": 0.832,
            "baseline_noun_f1": 0.316
        },
        "stage2_asc": {
            "name": "Aspect Sentiment Classification (ASC)",
            "architecture": "Position-Aware BiLSTM with Relative Distance-Decay Attention (PA-BiLSTM)",
            "improvements_from_scratch": [
                {
                    "issue": "Cross-Clause Sentiment Bleed",
                    "mechanism": "Distance-decay attention penalty: penalty_i = -alpha * |i - mid_aspect| (alpha=0.18)",
                    "result": "Prevents opposite sentiments in contrastive clauses ('screen is gorgeous but battery dies') from corrupting aspect attention."
                },
                {
                    "issue": "Aspect Bias via Unfrozen Embeddings",
                    "mechanism": "Frozen GloVe 100d Word Vectors + Learnable Position/Aspect Projections",
                    "result": "Prevents frequent aspect terms like 'build quality' or 'price' from drifting toward positive polarity in small training corpora."
                },
                {
                    "issue": "Coordinate Loss at Inference",
                    "mechanism": "Explicit relative crop offsets preserved and passed to tensor inputs",
                    "result": "Guarantees exact aspect-token grounding during real-time user inference."
                }
            ],
            "context_window": "k = 10 words (up to 40 tokens)",
            "word_embedding": "GloVe 6B 100-dimensional (Frozen)",
            "hidden_dim": 128,
            "attention_penalty_alpha": 0.18,
            "optimizer": "Adam (lr=1e-3, weight_decay=1e-5)",
            "loss_function": "Inverse-Frequency Class-Weighted Cross-Entropy Loss"
        }
    }


@app.get("/api/insights")
def get_insights():
    return {
        "categories": [
            {"name": "Screen / Display", "count": 33, "accuracy": 0.697, "positive_pct": 78, "negative_pct": 18, "neutral_pct": 4},
            {"name": "Battery / Power", "count": 19, "accuracy": 1.000, "positive_pct": 42, "negative_pct": 58, "neutral_pct": 0},
            {"name": "Keyboard / Input", "count": 30, "accuracy": 0.633, "positive_pct": 65, "negative_pct": 27, "neutral_pct": 8},
            {"name": "Design / Hardware", "count": 46, "accuracy": 0.609, "positive_pct": 68, "negative_pct": 25, "neutral_pct": 7},
            {"name": "Software / OS", "count": 60, "accuracy": 0.550, "positive_pct": 52, "negative_pct": 36, "neutral_pct": 12},
            {"name": "Performance / Speed", "count": 25, "accuracy": 0.560, "positive_pct": 64, "negative_pct": 28, "neutral_pct": 8},
            {"name": "Features / Usability", "count": 22, "accuracy": 0.727, "positive_pct": 73, "negative_pct": 18, "neutral_pct": 9},
            {"name": "Support / Service", "count": 23, "accuracy": 0.478, "positive_pct": 39, "negative_pct": 52, "neutral_pct": 9},
            {"name": "Price / Value", "count": 12, "accuracy": 0.667, "positive_pct": 50, "negative_pct": 50, "neutral_pct": 0}
        ],
        "top_aspects": [
            {"term": "battery life", "category": "battery/power", "frequency": 82, "sentiment": "Mixed"},
            {"term": "screen", "category": "screen/display", "frequency": 74, "sentiment": "Predominantly Positive"},
            {"term": "keyboard", "category": "keyboard/input", "frequency": 59, "sentiment": "Predominantly Positive"},
            {"term": "price", "category": "price/value", "frequency": 48, "sentiment": "Neutral / Balanced"},
            {"term": "graphics", "category": "screen/display", "frequency": 35, "sentiment": "Positive"},
            {"term": "windows 8", "category": "software/OS", "frequency": 32, "sentiment": "Mixed"},
            {"term": "trackpad", "category": "keyboard/input", "frequency": 28, "sentiment": "Predominantly Negative"}
        ]
    }


@app.get("/api/figures/{figure_name}")
def get_figure(figure_name: str):
    fig_path = FIGURES_DIR / figure_name
    if not fig_path.exists():
        raise HTTPException(status_code=404, detail="Figure not found.")
    return FileResponse(str(fig_path), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
