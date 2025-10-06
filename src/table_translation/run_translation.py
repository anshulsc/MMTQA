import time
from tqdm import tqdm
from pathlib import Path

from src.configs import translation_config as cfg
from src.table_translation.models.vLLM_client import VLLMClient
from src.table_translation.models.gemini_client import GeminiClient
from src.table_translation.translator import TableTranslator

def main():
    start_time = time.time()
    print("======================================================")
    print("      STARTING PHASE 1.2: TABLE TRANSLATION PIPELINE   ")
    print("======================================================")

    # 1. Initialize model clients
    vllm_client = VLLMClient(
        base_url=cfg.VLLM_BASE_URL,
        api_key=cfg.VLLM_API_KEY,
        model_name=cfg.VLLM_MODEL_NAME
    )
    gemini_client = GeminiClient(
        api_key=cfg.GEMINI_API_KEY,
        model_name=cfg.GEMINI_MODEL_NAME
    )

    # 2. Get list of source tables
    source_table_paths = sorted(list(cfg.TABLES_DIR.glob("*.json")))
    print(f"Found {len(source_table_paths)} source tables to translate.")

    # 3. Main loop over tables and languages
    for table_path in tqdm(source_table_paths, desc="Translating Tables"):
        table_id = table_path.stem
        print(f"\nProcessing table: {table_id}")
        
        translator = TableTranslator(table_id, table_path, vllm_client, gemini_client)
        
        for lang_code, lang_name in tqdm(cfg.LANGUAGES.items(), desc=f"Languages for {table_id}", leave=False):
            try:
                translator.run_for_language(lang_code, lang_name)
            except Exception as e:
                print(f"  [ERROR] An unexpected error occurred for language {lang_code}: {e}")

    end_time = time.time()
    print("\n======================================================")
    print("      PHASE 1.2: TRANSLATION PIPELINE COMPLETE      ")
    print(f"      Total time taken: {end_time - start_time:.2f} seconds")
    print("======================================================")

if __name__ == '__main__':
    main()