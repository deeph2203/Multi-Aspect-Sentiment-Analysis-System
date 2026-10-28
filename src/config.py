# src/config.py - Centralized Configuration for the ABSA Pipeline
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUTS_DIR = BASE_DIR / 'outputs'
MODELS_DIR = OUTPUTS_DIR / 'models'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
TABLES_DIR = OUTPUTS_DIR / 'tables'
DOCS_DIR = BASE_DIR / 'docs'

# Ensure required output directories exist
for p in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR, DOCS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Glove file path
GLOVE_FILE = DATA_DIR / 'glove.6B.100d.txt'

# Label mappings
LABELS = ['negative', 'neutral', 'positive']
LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
TAG2ID = {'O': 0, 'B-ASP': 1, 'I-ASP': 2}
ID2TAG = {v: k for k, v in TAG2ID.items()}

# Common limits & hyperparameters
SEED = 42
MAX_LEN = 60      # max sentence tokens
MAX_ASP = 5       # max aspect tokens
MAX_CTX = 20      # max context window tokens
K = 5             # context window size (+-5 tokens around aspect)

# Model architecture dimensions
W_DIM = 100       # word embedding dimension
P_DIM = 25        # POS tag / relative position dimension
HID = 128         # LSTM hidden units
DROP = 0.5        # dropout rate
ALPHA_DECAY = 0.18 # distance decay factor for position-aware attention

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
