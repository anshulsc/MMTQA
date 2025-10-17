import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


INPUT_QA_DIR = PROCESSED_DATA_DIR / "qa_pairs"
TABLES_DIR = PROCESSED_DATA_DIR / "tables" 

OUTPUT_DIR = PROCESSED_DATA_DIR / "qa_pairs_translated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUESTS_PER_MINUTE = 10 


LANGUAGES = {
    # Afro-Asiatic
    "ar": "Arabic (MSA)",

    # Austronesian
    "id_casual": "Indonesian (Casual)",
    "id_formal": "Indonesian (Formal)",
    "jv_krama": "Javanese (Krama - Polite)",
    "jv_ngoko": "Javanese (Ngoko - Casual)",
    "su_loma": "Sundanese",
    "tl": "Tagalog",

    # Indo-European
    "bn": "Bengali",
    "cs": "Czech",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "mr": "Marathi",
    "ru_formal": "Russian (Formal)",
    "sc": "Sardinian",
    "si_formal_spoken": "Sinhala",

    # Japonic
    "ja_formal": "Japanese (Formal)",

    # Koreanic
    "ko_formal": "Korean (Formal)",

    # Kra-Dai
    "th": "Thai",


    # Sino-Tibetan
    "nan": "Hokkien (Written)",
    "zh_cn": "Chinese (Mandarin)",

    # Turkic
    "az": "Azerbaijani"
}


GEMINI_API_KEYS = [
"API"
]
GEMINI_MODEL_NAME = "gemini-2.5-flash"