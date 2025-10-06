import json
import pandas as pd
from hashlib import sha256

def save_table(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def save_metadata(metadata: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

def get_table_hash(df: pd.DataFrame) -> str:
    return sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()