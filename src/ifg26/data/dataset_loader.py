"""
dataset_loader.py

Standard functions for loading and merging IFG-26 triplet and decoy datasets.
"""
import pandas as pd
import pathlib

def load_triplet_dataset(path: str) -> pd.DataFrame:
    """Load the positive triplets."""
    # Stub reading from parquet or csv
    # The actual implementation depends on what extensions are used
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def load_decoy_pool(path: str) -> pd.DataFrame:
    """Load the decoy pool."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def merge_datasets(positives: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    """Merge positives and negatives and format for modeling."""
    
    # ensure label columns
    if 'label' not in positives.columns:
        positives['label'] = 1
    if 'label' not in negatives.columns:
        negatives['label'] = 0
        
    df = pd.concat([positives, negatives], ignore_index=True)
    return df

if __name__ == "__main__":
    print("Smoke Test for dataset_loader.py")
    p = pd.DataFrame({"ligand_smiles": ["C"], "label": [1]})
    n = pd.DataFrame({"ligand_smiles": ["CC"], "label": [0]})
    df = merge_datasets(p, n)
    assert len(df) == 2 and "label" in df.columns
    print("Passed!")
