import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from termcolor import cprint

def load_benchmark_data(data_file_path: str, image_type: str, lang_code_filter: str) -> List[Dict]:

    data_file = Path(data_file_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found at: {data_file_path}")
        
    evaluation_set = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cprint(f"Reading {len(lines)} lines from {data_file.name}...", "cyan")
    for line in tqdm(lines, desc="Loading data"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        row_lang = row.get("language", "en")
        if lang_code_filter != "default" and row_lang != lang_code_filter:
            continue
        
        # In a consolidated dataset, the image filenames are already provided.
        # We just need to pick the right list of images.
        image_list_key = 'image_clean' if image_type == 'clean' else 'image_noise'
        image_filenames = row.get(image_list_key, [])

        if not image_filenames:
            continue

        # Create an evaluation instance for EACH available image (clean or noisy)
        for image_filename in image_filenames:
            instance = {
                "question_id": row.get("question_id"),
                "table_id": row.get("table_id"),
                "language": row_lang,
                "question": row.get("question"),
                "golden_answer": row.get("answer"),
                "reasoning_category": row.get("reasoning_category"),
                "question_type": row.get("question_type"),
                "image_filename": image_filename,
            }
            evaluation_set.append(instance)
            
    if lang_code_filter != "default":
        cprint(f"Filtered for language '{lang_code_filter}'.", "green")
        
    cprint(f"Created a total of {len(evaluation_set)} evaluation instances using '{image_type}' images.", "green")
    return evaluation_set