import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm
from pathlib import Path

from src.configs import collection_config as cfg
from termcolor import cprint
from src.utils.file_utils import save_table, save_metadata, get_table_hash

class BaseCollector(ABC):
    
    def __init__(self, source_name: str, target_count: int):
        self.source_name = source_name
        self.target_count = target_count
        self.save_table_dir = cfg.TABLES_DIR
        self.save_meta_dir = cfg.METADATA_DIR
        self.collected_count = 0
        self.pbar = None

    @abstractmethod
    def collect(self):
        raise NotImplementedError("Subclasses must implement the collect method.")

    def _apply_quality_filters(self, df: pd.DataFrame) -> bool:
        if df.shape[0] < cfg.MIN_ROWS or df.shape[1] < cfg.MIN_COLS:
            return False
        
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > cfg.MAX_MISSING_RATIO:
            return False
            
        return True

    def _process_and_save(self, df: pd.DataFrame, extra_metadata: dict = None):
        if not self._apply_quality_filters(df):
            return

        table_hash = get_table_hash(df)
        table_filename = f"{self.source_name}_{table_hash[:10]}.csv"
        meta_filename = f"{self.source_name}_{table_hash[:10]}.json"
        
        table_path = self.save_table_dir / table_filename
        meta_path = self.save_meta_dir / meta_filename

        if table_path.exists():
            return

        metadata = {
            "table_id": f"{self.source_name}_{table_hash[:10]}",
            "source": self.source_name,
            "original_metadata": extra_metadata or {},
            "dimensions": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            },
            "schema": {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        }

        save_table(df, table_path)
        save_metadata(metadata, meta_path)
        
        self.collected_count += 1
        if self.pbar:
            self.pbar.update(1)

    def run(self):
        cprint(f"--- Starting collection from {self.source_name} ---", color="cyan")
        self.pbar = tqdm(total=self.target_count, desc=f"Collecting from {self.source_name}")
        self.collect()
        self.pbar.close()
        cprint(f"--- Finished collection from {self.source_name}. Collected {self.collected_count} tables. ---\n", color="green")