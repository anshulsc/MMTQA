import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from termcolor import cprint

def load_benchmark_data(
    data_file_path: str,
    images_root_dir: str,
    image_type: str,
    lang_code_filter: str
) -> List[Dict]:
    """
    Loads benchmark data by pairing QA entries from a JSONL file with image files
    discovered on the disk based on table_id and language.

    Args:
        data_file_path: Path to the .jsonl file with QA pairs.
        images_root_dir: Path to the root 'images' directory.
        image_type: The subfolder to search within ('clean' or 'noise').
        lang_code_filter: The language code to filter by, or 'default'.

    Returns:
        A list of dictionaries for the evaluation set.
    """
    data_file = Path(data_file_path)
    images_root = Path(images_root_dir)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root directory not found: {images_root_dir}")

    evaluation_set = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cprint(f"Processing {len(lines)} QA entries from {data_file.name}...", "cyan")
    for line in tqdm(lines, desc="Matching QA with images"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        # --- LANGUAGE FILTERING ---
        row_lang = row.get("language", "en")
        if lang_code_filter != "default" and row_lang != lang_code_filter:
            continue
        
        # --- DYNAMIC IMAGE DISCOVERY ---
        table_id = row.get("table_id")
        if not table_id:
            continue

        # 1. Construct the target directory path for the images
        target_image_dir = images_root / table_id / image_type
        
        if not target_image_dir.is_dir():
            continue # Skip if the corresponding image folder doesn't exist

        # 2. Find all images in that directory matching the language code
        #    Pattern: en_clean.jpg, en_noise1.jpg, en_noise2.jpg, etc.
        image_search_pattern = f"{row_lang}_*.jpg"
        found_images = sorted(list(target_image_dir.glob(image_search_pattern)))

        # 3. Create an evaluation instance for each discovered image
        for image_path in found_images:
            instance = {
                "question_id": row.get("question_id"),
                "table_id": table_id,
                "language": row_lang,
                "question": row.get("question"),
                "golden_answer": row.get("answer"),
                "reasoning_category": row.get("reasoning_category"),
                "question_type": row.get("question_type"),
                "image_filename": image_path.name, # Store just the filename
            }
            evaluation_set.append(instance)
            
    if lang_code_filter != "default":
        cprint(f"Filtered for language '{lang_code_filter}'.", "green")
        
    cprint(f"Created a total of {len(evaluation_set)} evaluation instances using '{image_type}' images.", "green")
    return evaluation_set