import os
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = PROCESSED_DATA_DIR / "tables"  


TRANSLATED_TABLES_DIR = PROCESSED_DATA_DIR / "tables_translated"
CHECKPOINTS_DIR = PROCESSED_DATA_DIR / "translation_checkpoints"
TRANSLATION_METADATA_DIR = PROCESSED_DATA_DIR / "translation_metadata"

TRANSLATED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATION_METADATA_DIR.mkdir(parents=True, exist_ok=True)



SOURCE_LANG_CODE = "en"
SOURCE_LANG_NAME = "English"

LANGUAGES = {
    # Afro-Asiatic
    # "ar": "Arabic (MSA)",

    # Austronesian
    # "id_casual": "Indonesian (Casual)",
    # "id_formal": "Indonesian (Formal)",
    # "jv_krama": "Javanese (Krama - Polite)",
    # "jv_ngoko": "Javanese (Ngoko - Casual)",
    # "su_loma": "Sundanese",
    # "tl": "Tagalog",

    # Indo-European
    # "bn": "Bengali",
    # "cs": "Czech",
    # "en": "English",
    # "es": "Spanish",
    # "fr": "French",
    # "hi": "Hindi",
    # "it": "Italian",
    # "mr": "Marathi",
    # "ru_casual": "Russian (Casual)",
    # "ru_formal": "Russian (Formal)",
    "sc": "Sardinian",
    "si_formal_spoken": "Sinhala",

    # Japonic
    "ja_casual": "Japanese (Casual)",
    "ja_formal": "Japanese (Formal)",

    # Koreanic
    "ko_casual": "Korean (Casual)",
    "ko_formal": "Korean (Formal)",

    # Kra-Dai
    "th": "Thai",

    # Niger-Congo
    "yo": "Yoruba",

    # Sino-Tibetan
    "nan": "Hokkien (Written)",
    "nan_spoken": "Hokkien (Spoken)",
    "yue": "Cantonese",
    "zh_cn": "Chinese (Mandarin)",

    # Turkic
    "az": "Azerbaijani"
}




VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct" 

GEMINI_API_KEY = "EMPTY"
GEMINI_MODEL_NAME = "gemini-2.5-flash"



BLEU_THRESHOLD = 0.40 