import pandas as pd
from datasets import load_dataset
from typing import List, Dict, Any

from src.data_collection.base_collector import BaseCollector
from src.configs import collection_config as cfg

class FinqaCollector(BaseCollector):
    def __init__(self, target_count: int):
        super().__init__(source_name="finqa", target_count=target_count)
        self.raw_dataset = None

    def _load_dataset(self):
        self.raw_dataset = load_dataset("dreamerdeo/finqa", cache_dir=cfg.RAW_DATA_DIR, trust_remote_code=True)["train"]
        print(f"Loaded {len(self.raw_dataset)} FinQA examples.")

    def collect(self):
        if self.raw_dataset is None:
            self._load_dataset()
            
        for item in self.raw_dataset:
            if self.collected_count >= self.target_count:
                break
            
            try:
                
                table_data_raw: List[List[str]] = item['table']
                
                if not table_data_raw or len(table_data_raw) < 2:
                    continue
                
                header = table_data_raw[0]
                
                rows = table_data_raw[1:]

                df = pd.DataFrame(rows, columns=header)
                
                extra_meta = {
                    "source_dataset": "finqa",
                    "document_id": item['id'],
                    "pre_text_caption_evidence": item['pre_text'],
                    "question": item['question'],
                }

                self._process_and_save(df, extra_meta)
                
            except Exception as e:
                continue

if __name__ == '__main__':
    print("Running FinqaCollector as a standalone script...")
    target = cfg.COLLECTION_TARGETS.get("finqa", 50) 
    collector = FinqaCollector(target_count=target)
    collector.run()
    print("Standalone run finished.")