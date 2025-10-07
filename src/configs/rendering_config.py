from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


TABLES_DIR = PROCESSED_DATA_DIR / "tables"
TRANSLATED_TABLES_DIR = PROCESSED_DATA_DIR / "tables_translated"

VISUAL_IMAGES_DIR = PROCESSED_DATA_DIR / "visual_images"
VISUAL_METADATA_DIR = PROCESSED_DATA_DIR / "visual_metadata"

VISUAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_METADATA_DIR.mkdir(parents=True, exist_ok=True)


NUM_VERSIONS_PER_TABLE = 3  
BASE_FONT_SIZE = "14px"    
BASE_WIDTH = 1200           