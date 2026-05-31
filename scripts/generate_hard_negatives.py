"""
generate_hard_negatives.py
=========================
Generates the HARD-NEAR negative tier focusing on chemical similarity 
and property matching.
"""

import os
import pandas as pd
from pathlib import Path
from ifg26.splits.hard_negatives import build_hard_near_tier

ROOT = Path(r"C:\Users\Aadi Nair\Downloads\IFG26")
POS_PATH = ROOT / "data/pu/pool_P_scaffold.parquet"
UNIVERSE_PATH = ROOT / "data/external_chembl_universe.parquet"
OUT_PATH = ROOT / "data/phase5_hard_near_negatives.parquet"

def main():
    if not POS_PATH.exists() or not UNIVERSE_PATH.exists():
        print("Missing input files. Please ensure you have data/pu/pool_P_scaffold.parquet and data/external_chembl_universe.parquet.")
        return

    print("Loading positive training pool...")
    pos_df = pd.read_parquet(POS_PATH)
    # Filter for unique ligands to reduce work
    unique_pos = pos_df.drop_duplicates(subset=['ligand_smiles'])
    print(f"Unique positive ligands: {len(unique_pos)}")

    print("Loading external universe (e.g. ChEMBL subset)...")
    universe_df = pd.read_parquet(UNIVERSE_PATH)
    # Rename columns to match what the internal function expects
    universe_df = universe_df.rename(columns={'MolWt': 'mw', 'MolLogP': 'logp'})
    print(f"Candidates available: {len(universe_df)}")

    print("Building HARD-NEAR negatives tier...")
    hard_near_df = build_hard_near_tier(unique_pos.head(200), universe_df, n_per_pos=10) # Using subset for demo speed
    
    print(f"Generated {len(hard_near_df)} hard-near negatives.")
    print(hard_near_df.head())
    
    hard_near_df.to_parquet(OUT_PATH, index=False)
    print(f"Hard-near negatives saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
