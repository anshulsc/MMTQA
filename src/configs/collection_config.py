import os
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TABLES_DIR = PROCESSED_DATA_DIR / "tables"
METADATA_DIR = PROCESSED_DATA_DIR / "metadata"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


COLLECTION_TARGETS = {
    "wikisql": 5000,
    "pubtabnet": 3000,
    "arxiv": 2000,
}


MIN_ROWS = 3
MIN_COLS = 3
MAX_MISSING_RATIO = 0.8  