# Multi-Aspect Sentiment Analysis System for Product Reviews

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![React Vite](https://img.shields.io/badge/Frontend-React%20%7C%20Vite-61dafb.svg)](https://vitejs.dev/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, high-performance Two-Stage **Aspect-Based Sentiment Analysis (ABSA)** deep learning platform built on the SemEval-2014 Task 4 (Laptop Reviews) benchmark. The system decouples **Aspect Term Extraction (ATE)** via a Bidirectional LSTM-CRF sequence tagger from **Aspect Sentiment Classification (ASC)** via a custom **Position-Aware Distance-Decay Attention BiLSTM (PA-BiLSTM)**.

---

## 🌟 Key Highlights & Engineering Breakthroughs

1. **Solved Cross-Clause Attention Bleed**:
   - *Problem*: Traditional attention models leak sentiment across contrastive clauses (*"The screen is gorgeous but the battery dies in two hours"* marked screen as negative).
   - *Solution*: Implemented a relative distance-decay attention mechanism ($\text{penalty}_i = -\alpha \cdot |i - \text{mid}_{\text{aspect}}|$, with $\alpha=0.18$), forcing the model to attend strictly to the aspect's governing clause.
   - *Result*: Contrastive test pass rate jumped from **0% on baseline to 60%**.

2. **Eliminated Aspect Word Bias**:
   - Frozen GloVe 100d embeddings prevent common aspect terms (e.g., *"build quality"*, *"price"*) from drifting into positive polarity, correctly classifying negative modifiers (*"The build quality is terrible"* → **negative**).

3. **High-Performance Production Stack**:
   - **Modern React + Vite Frontend**: Beautiful dark glassmorphic UI with 4 dedicated pages, real-time token highlighting, confidence meters, probability distributions, benchmark comparison matrices, and EDA charts.
   - **FastAPI ASGI Backend**: Real-time CPU inference engine with sub-50ms latency.

---

## 🚀 Quickstart & Localhost Execution

### 1. Prerequisites & Environment Setup
```bash
# Clone the repository
git clone https://github.com/deeph2203/Multi-Aspect-Sentiment-Analysis-System.git
cd Multi-Aspect-Sentiment-Analysis-System

# Install Python backend dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Launch FastAPI Backend Server
```bash
# Starts ASGI backend on http://127.0.0.1:8000
python backend/server.py
```
*API interactive documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).*

### 3. Launch React Frontend
```bash
cd frontend
npm install
npm run dev
```
*Open your browser at **[http://localhost:5173](http://localhost:5173)** to use the application.*

---

## 🖥️ Web Application Structure (4 Dedicated Pages)

| Page | Path / Feature | Description |
| :--- | :--- | :--- |
| **1. Live Aspect Analyzer** | `/` (Main Page) | Full end-to-end inference engine: real-time review input, model selector (**Position-Aware BiLSTM** vs **Baseline BiLSTM**), 6 benchmark preset review chips, interactive token grounding, class probability distribution bars, confidence gauges, and attention peak keywords. |
| **2. Accuracy & Benchmarks** | Tab: *Accuracy & Benchmarks* | Hero KPI cards (+134% ATE gain, 64.1% ASC accuracy), comprehensive 9-model comparison leaderboard (VADER, SVM, CNN, BiLSTM, BERT), 10-category granular accuracy breakdown, filterable 24-case diagnostic test suite, and confusion matrix viewer. |
| **3. Architecture & Pipeline** | Tab: *Architecture & Pipeline* | Interactive visual walkthrough of Stage 1 (BiLSTM-CRF) and Stage 2 (PA-BiLSTM-Attn), failure mode diagnostic case studies, mathematical formulations of distance decay, and exact tensor hyperparameters. |
| **4. Dataset & Insights** | Tab: *Dataset & Insights* | SemEval-2014 dataset statistics, class balance breakdown, high-resolution EDA figures gallery, stacked category polarity charts, and business intelligence implications for e-commerce. |

---

## 📊 Benchmark Results

### Stage 1: Aspect Term Extraction (ATE) — Test Set
| Model Architecture | Exact Precision | Exact Recall | Exact F1 | Partial F1 | Gain over Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Rule Baseline (POS Nouns) | 0.202 | 0.723 | 0.316 | 0.405 | — |
| **BiLSTM-CRF (Ours)** | **0.738** | **0.746** | **0.742** | **0.832** | **+134.8%** |

### Stage 2: Aspect Sentiment Classification (ASC) — Test Set
| Model Architecture | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Gain over Rule Baseline |
| :--- | :---: | :---: | :---: | :---: |
| Rule-based (VADER) | 0.494 | 49.9% | 0.475 | 0.0% |
| Naive Bayes | 0.606 | 63.6% | 0.565 | +19.0% |
| Linear SVM | 0.621 | 65.6% | 0.595 | +25.3% |
| Logistic Regression | 0.608 | 64.7% | 0.608 | +28.2% |
| CNN (Aspect Window) | 0.684 | 66.6% | 0.626 | +31.9% |
| BiLSTM (No Aspect) | 0.689 | 62.4% | 0.580 | +22.3% |
| BiLSTM + Aspect Attention (Baseline) | 0.638 | 60.4% | 0.565 | +19.0% |
| **Position-Aware BiLSTM (Trained & Optimized)** | **0.647** | **64.1%** | **0.617** | **+29.9%** |
| BERT (Sentence + Aspect Reference) | 0.738 | 74.3% | 0.716 | +50.9% |

---

## 📁 Modular Codebase Architecture

```
Multi-Aspect-Sentiment-Analysis-System/
│
├── backend/
│   └── server.py                  # Production FastAPI server (endpoints for analyze, benchmarks, pipeline, figures)
│
├── frontend/                      # Modern React + Vite application
│   ├── src/
│   │   ├── components/Navbar.jsx  # Navigation bar with live engine status & links
│   │   ├── pages/
│   │   │   ├── AnalyzerPage.jsx   # Live aspect extraction & sentiment analyzer
│   │   │   ├── BenchmarksPage.jsx # Accuracy cards, leaderboard, 24 test cases, confusion matrices
│   │   │   ├── PipelinePage.jsx   # Architecture flowchart, failure mode fixes, hyperparameters
│   │   │   └── InsightsPage.jsx   # EDA charts, category polarity proportions, business insights
│   │   ├── App.jsx                # Main application state & API coordinator
│   │   └── index.css              # Dark-mode glassmorphic CSS design system
│   ├── package.json
│   └── vite.config.js             # Configured with proxy to port 8000
│
├── src/                           # Core reusable Python NLP package
│   ├── config.py                  # Hyperparameters, directories, label definitions
│   ├── data/                      # Parser, preprocessor, BIO tagger
│   ├── features/                  # Context window builder, GloVe vectors
│   ├── models/                    # BiLSTM-CRF, Position-Aware BiLSTM, Baseline BiLSTM
│   ├── evaluation/                # Metrics, category mapping, error analysis
│   └── inference/                 # AspectAnalyzer end-to-end pipeline
│
├── outputs/                       # Pre-trained model weights & visualizations
│   ├── models/                    # ate_bilstm_crf.pt, asc_position_aware.pt, asc_bilstm_attn.pt
│   ├── figures/                   # EDA overview, confusion matrices, category charts
│   └── tables/                    # Benchmark comparison CSVs, test predictions
│
├── scripts/                       # Modular pipeline runners (Step 1 through Step 6)
├── tests/                         # 24-case diagnostic test suite (tests/test_pipeline.py)
├── docs/                          # Technical specifications, reports, and data tables
├── inference.py                   # Root Python convenience entrypoint
├── requirements.txt               # Pinned backend dependencies
└── README.md
```

---

## 🛠️ Automated Testing
To run the automated 24-case benchmark test suite across all 7 challenge categories:
```bash
python tests/test_pipeline.py
```

---

## 👥 Authors & Academic Context
Developed for NLP Project (Sem 7) on Aspect-Based Sentiment Analysis on the SemEval-2014 Task 4 benchmark dataset.
