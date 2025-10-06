# src/config.py
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- Data Collection Settings ---
COLLECTION_TARGETS = {
    "wikitables": 5000,
    "pubtabnet": 5000,
    "arxiv": 2000,
}


MIN_ROWS = 3
MIN_COLS = 3
MAX_MISSING_VAL_RATIO = 0.8

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)