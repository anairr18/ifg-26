"""
generate_protein_embeddings.py
==============================
Canonical pipeline for extracting ESM2 protein embeddings.
Uses the ifg26.features.protein_embeddings module.
"""

import os
import argparse
import sys
import time
import urllib.request
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from ifg26.features.protein_embeddings import get_cached_embeddings

ROOT = Path(r"C:\Users\Aadi Nair\Downloads\IFG26")
P_IDX_PATH = ROOT / "data/features/protein_index.parquet"
P_SEQ_PATH = ROOT / "data/features/protein_sequences.parquet"
CACHE_PATH = ROOT / "data/features/protein_embeddings_esm2_650m.parquet"

def fetch_sequence(uniprot_id: str) -> str | None:
    """Fetch canonical FASTA sequence from UniProt REST API."""
    uid = uniprot_id.split("_")[0].strip()
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            fasta = resp.read().decode("utf-8")
        lines = fasta.strip().split("\n")
        seq = "".join(lines[1:])
        return seq
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="esm2_t33_650M_UR50D", help="ESM2 model to use")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for embedding")
    args = parser.parse_args()

    if not P_IDX_PATH.exists():
        print(f"Error: Protein index not found at {P_IDX_PATH}")
        sys.exit(1)

    p_idx = pd.read_parquet(P_IDX_PATH)
    uids = p_idx["uniprot_id"].unique().tolist()
    print(f"Dataset unique proteins: {len(uids)}")

    # Load or fetch sequences
    if P_SEQ_PATH.exists():
        seq_df = pd.read_parquet(P_SEQ_PATH)
        print(f"Loaded {len(seq_df)} sequences from cache.")
    else:
        seq_df = pd.DataFrame(columns=["uniprot_id", "sequence"])

    missing_uids = [u for u in uids if u not in seq_df["uniprot_id"].tolist()]
    if missing_uids:
        print(f"Fetching {len(missing_uids)} missing sequences from UniProt...")
        new_seqs = []
        for uid in missing_uids:
            seq = fetch_sequence(uid)
            if seq:
                new_seqs.append({"uniprot_id": uid, "sequence": seq})
                print(f"  Fetched {uid} ({len(seq)} aa)")
            else:
                print(f"  Failed {uid}")
            time.sleep(0.1) # Be nice
        
        seq_df = pd.concat([seq_df, pd.DataFrame(new_seqs)], ignore_index=True)
        seq_df.to_parquet(P_SEQ_PATH, index=False)

    # Embed
    merged_with_meta = seq_df[seq_df["uniprot_id"].isin(uids)]
    print(f"Starting ESM2 extraction for {len(merged_with_meta)} proteins...")
    
    results = get_cached_embeddings(
        uniprot_ids=merged_with_meta["uniprot_id"].tolist(),
        sequences=merged_with_meta["sequence"].tolist(),
        cache_path=CACHE_PATH,
        model_name=args.model
    )

    print(f"Extraction complete. Embeddings saved to: {CACHE_PATH}")
    print(f"Embedding dimension: {len(results['embedding'].iloc[0])}")

if __name__ == "__main__":
    main()
