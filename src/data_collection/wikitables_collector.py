import pandas as pd
from datasets import load_dataset

from src.data_collection.base_collector import BaseCollector
from src.configs import collection_config as cfg

class WikiTableCollector(BaseCollector):
    def __init__(self, target_count: int):
        super().__init__(source_name="wikisql", target_count=target_count)

    def collect(self):
        dataset = load_dataset(
            "wikisql",
            split="train",
            cache_dir=cfg.RAW_DATA_DIR,
            trust_remote_code=True,
        )
        
        for item in dataset:
            if self.collected_count >= self.target_count:
                break

            try:
                header = item['table']['header']
                rows = item['table']['rows']
                df = pd.DataFrame(rows, columns=header)
                
                extra_meta = {
                    "source_dataset": "wikisql",
                    "table_page_title": item['table']['page_title']
                }
                
                self._process_and_save(df, extra_meta)
            except Exception as e:
                continue

if __name__ == '__main__':
    print("Running WikiTableCollector as a standalone script...")
    target = cfg.COLLECTION_TARGETS.get("wikisql", 100) 
    collector = WikiTableCollector(target_count=target)
    collector.run()
    print("Standalone run finished.")