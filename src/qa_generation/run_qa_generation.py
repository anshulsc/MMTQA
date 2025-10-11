import json
import time
from tqdm import tqdm

from src.configs import qa_config as cfg
from src.qa_generation.qa_generator import QAGenerator

def main():
    start_time = time.time()
    print("======================================================")
    print("      STARTING PHASE 2: QA GENERATION (ENGLISH)      ")
    print("======================================================")


    source_table_paths = sorted(list(cfg.TABLES_DIR.glob("*.json")))
    print(f"Found {len(source_table_paths)} source English tables.")

    for table_path in tqdm(source_table_paths, desc="Generating QA Pairs"):
        table_id = table_path.stem
        output_path = cfg.QA_PAIRS_DIR / f"{table_id}_qa.json"

        if output_path.exists():
            tqdm.write(f"Skipping {table_id}, QA file already exists.")
            continue
            
        tqdm.write(f"\nProcessing table: {table_id}")
        
        try:
            with open(table_path, 'r', encoding='utf-8') as f:
                table_data = json.load(f)
        except json.JSONDecodeError:
            tqdm.write(f"  [WARN] Skipping malformed JSON file: {table_path}")
            continue

        generator = QAGenerator(table_id, table_data)
        qa_collection = generator.generate()

        if qa_collection:
            final_output = []
            for i, qa_pair in enumerate(qa_collection.qa_pairs):
                final_item = {
                    "question_id": f"{table_id}_{i+1:03d}",
                    "table_id": table_id,
                    "question_type": qa_pair.question_type,
                    "question": qa_pair.question,
                    "answer": qa_pair.answer,
                    "evidence_cells": qa_pair.evidence_cells,
                    "reasoning_category": qa_pair.reasoning_category
                }
                final_output.append(final_item)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2)
            tqdm.write(f"  Successfully saved {len(final_output)} QA pairs to {output_path.name}")

    end_time = time.time()
    print("\n======================================================")
    print("      PHASE 2: QA GENERATION COMPLETE      ")
    print(f"      Total time taken: {end_time - start_time:.2f} seconds")
    print("======================================================")

if __name__ == '__main__':
    main()