import json
import time
from tqdm import tqdm
from pathlib import Path

from src.configs import rendering_config as cfg
from src.rendering.renderer import TableRenderer

def main():
    start_time = time.time()
    print("======================================================")
    print("      STARTING PHASE 1.3: VISUAL TABLE RENDERING      ")
    print("======================================================")

  
    source_tables = list(cfg.TABLES_DIR.glob("*.json"))
    translated_tables = list(cfg.TRANSLATED_TABLES_DIR.glob("**/*.json"))
    
    all_tables_to_process = []
    
    for path in source_tables:
        all_tables_to_process.append({"path": path, "lang": "en"})
    
   
    for path in translated_tables:
        lang_code = path.stem
        source_table_id = path.parent.name
     
        original_table_path = cfg.TRANSLATED_TABLES_DIR / source_table_id / f"{lang_code}.json"
        if original_table_path.exists():
             all_tables_to_process.append({"path": original_table_path, "lang": lang_code})

    print(f"Found {len(all_tables_to_process)} total table instances to render.")
    
    
    for item in tqdm(all_tables_to_process, desc="Rendering Tables"):
        table_path = item['path']
        lang_code = item['lang']
        source_table_id = table_path.parent.name if 'translated' in str(table_path) else table_path.stem

        try:
            table_data = json.loads(table_path.read_text(encoding='utf-8'))
            
            for i in range(cfg.NUM_VERSIONS_PER_TABLE):
                renderer = TableRenderer(source_table_id, lang_code, table_data)
                renderer.render_and_save()

        except json.JSONDecodeError:
            print(f"  [WARN] Skipping malformed JSON file: {table_path}")
        except Exception as e:
            print(f"  [FATAL] Unexpected error processing {table_path}: {e}")
            
    end_time = time.time()
    print("\n======================================================")
    print("      PHASE 1.3: VISUAL RENDERING COMPLETE      ")
    print(f"      Total time taken: {end_time - start_time:.2f} seconds")
    print("======================================================")


if __name__ == '__main__':
    main()