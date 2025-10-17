import json
import time
from tqdm import tqdm
from pathlib import Path

from src.configs import qa_translation_config as cfg
from src.qa_translation.translator import QATranslator

def main():
    start_time = time.time()
    print("=========================================================")
    print("      STARTING PHASE 2.2: QA PAIR TRANSLATION          ")
    print("=========================================================")
    print(f"Rate limit: {getattr(cfg, 'REQUESTS_PER_MINUTE', 10)} requests/minute per key")
    print(f"Total API keys: {len(cfg.GEMINI_API_KEYS)}")
    print(f"Target languages: {len(cfg.LANGUAGES)}")
    print("=========================================================\n")

    english_qa_paths = sorted(list(cfg.INPUT_QA_DIR.glob("*.json")))
    print(f"Found {len(english_qa_paths)} English QA files to translate.\n")

    # Statistics
    stats = {
        "total_files": len(english_qa_paths) * len(cfg.LANGUAGES),
        "skipped": 0,
        "completed": 0,
        "failed": 0
    }

    for english_qa_path in tqdm(english_qa_paths, desc="Files"):
        table_id = english_qa_path.stem.replace("_qa", "")
        
        # Load the corresponding context table
        table_path = cfg.TABLES_DIR / f"{table_id}.json"
        if not table_path.exists():
            tqdm.write(f"  [WARN] Skipping {english_qa_path.name}, context table not found.")
            stats["failed"] += len(cfg.LANGUAGES)
            continue
        
        with open(table_path, 'r', encoding='utf-8') as f:
            context_table = json.load(f)

        # Load the English QA pairs for this table
        with open(english_qa_path, 'r', encoding='utf-8') as f:
            english_qa_list = json.load(f)

        for lang_code, lang_name in tqdm(cfg.LANGUAGES.items(), desc=f"Languages for {table_id}", leave=False):
            output_dir = cfg.OUTPUT_DIR / lang_code
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / english_qa_path.name

            # Skip if already completed
            if QATranslator.should_skip_translation(lang_code, table_id):
                tqdm.write(f"  [SKIP] {table_id} -> {lang_name} ({lang_code}) already completed.")
                stats["skipped"] += 1
                continue

            tqdm.write(f"\n  Translating {table_id} to {lang_name} ({lang_code})")
            
            translated_qa_list = []
            failed_qa_count = 0
            
            for idx, english_qa_pair in enumerate(tqdm(english_qa_list, desc=f"QA Pairs ({lang_code})", leave=False)):
                try:
                    translator = QATranslator(english_qa_pair, context_table)
                    
                    if lang_code=="en":
                            new_qa_pair = english_qa_pair.copy()
                            translated_qa_list.append(new_qa_pair)
                    else:
                            translation_result = translator.translate(lang_name)
                            
                            if translation_result:
                                # Construct the new translated QA pair, preserving original metadata
                                new_qa_pair = english_qa_pair.copy()
                                new_qa_pair['question'] = translation_result.translated_question
                                new_qa_pair['answer'] = translation_result.translated_answer
                                translated_qa_list.append(new_qa_pair)
                            else:
                                failed_qa_count += 1
                                tqdm.write(f"    [WARN] Failed to translate QA pair {idx + 1}")
                
                except Exception as e:
                    failed_qa_count += 1
                    tqdm.write(f"    [ERROR] Exception on QA pair {idx + 1}: {e}")
                    
                    # If all keys exhausted, stop processing this language
                    if "all api keys have exceeded" in str(e).lower():
                        tqdm.write(f"    [CRITICAL] Stopping translation due to quota exhaustion.")
                        break

            # Save translated QA pairs even if some failed
            if translated_qa_list:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(translated_qa_list, f, indent=2, ensure_ascii=False)
                
                # Mark as complete
                QATranslator.mark_translation_complete(lang_code, table_id)
                
                success_rate = len(translated_qa_list) / len(english_qa_list) * 100
                tqdm.write(f"    [SUCCESS] Saved {len(translated_qa_list)}/{len(english_qa_list)} "
                          f"({success_rate:.1f}%) QA pairs to {output_path}")
                
                if failed_qa_count > 0:
                    tqdm.write(f"    [WARN] {failed_qa_count} QA pairs failed translation")
                
                stats["completed"] += 1
            else:
                tqdm.write(f"    [FAIL] No QA pairs successfully translated for {lang_name}")
                stats["failed"] += 1

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    
    print("\n=========================================================")
    print("      PHASE 2.2: QA PAIR TRANSLATION COMPLETE        ")
    print("=========================================================")
    print(f"Total time: {elapsed_minutes:.2f} minutes")
    print(f"\nStatistics:")
    print(f"  Total tasks: {stats['total_files']}")
    print(f"  Completed: {stats['completed']} ({stats['completed']/stats['total_files']*100:.1f}%)")
    print(f"  Skipped: {stats['skipped']} ({stats['skipped']/stats['total_files']*100:.1f}%)")
    print(f"  Failed: {stats['failed']} ({stats['failed']/stats['total_files']*100:.1f}%)")
    print("=========================================================")


if __name__ == '__main__':
    main()