import json
import pandas as pd
from hashlib import sha256
from pathlib import Path


def df_to_structured_json(df: pd.DataFrame) -> dict:
    columns = df.columns.tolist()
    data_rows = df.values.tolist()
    
    return {
        "columns": columns,
        "data": data_rows
    }

def save_table_json(data: list[dict], path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_metadata(metadata: dict, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

def get_table_hash(df: pd.DataFrame) -> str:
    return sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()