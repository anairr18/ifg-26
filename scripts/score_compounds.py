"""
score_compounds.py
==================
CLI tool for high-throughput molecular glue scoring.
Usage: python scripts/score_compounds.py --model model.pt --smiles input.csv --e3 P03372 --target Q07869 --out results.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from ifg26.inference.scorer import IFG26Scorer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained DualModalityMLP state_dict")
    parser.add_argument("--smiles", required=True, help="CSV file with 'smiles' column")
    parser.add_argument("--e3", required=True, help="Protein sequence or UniProt ID (if pre-cached)")
    parser.add_argument("--target", required=True, help="Protein sequence or UniProt ID (if pre-cached)")
    parser.add_argument("--out", default="screening_results.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading IFG-26 Scorer from {args.model}...")
    # Note: In a real system, we'd handle ID-to-Sequence lookup. 
    # Here we assume sequence strings are provided.
    scorer = IFG26Scorer(args.model)

    print(f"Loading compounds from {args.smiles}...")
    df = pd.read_csv(args.smiles)
    if 'smiles' not in df.columns:
        if 'SMILES' in df.columns: df = df.rename(columns={'SMILES': 'smiles'})
        else: raise ValueError("CSV must contain 'smiles' column")

    smiles_list = df['smiles'].tolist()
    print(f"Scoring {len(smiles_list)} candidates...")
    
    scores = scorer.score(smiles_list, args.e3, args.target, batch_size=args.batch_size)
    
    df['ifg26_score'] = scores
    df = df.sort_values('ifg26_score', ascending=False)
    
    df.to_csv(args.out, index=False)
    print(f"Results saved to {args.out}")
    print(df.head(10))

if __name__ == "__main__":
    main()
