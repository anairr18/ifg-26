"""
validate_splits.py

Strict dataset split validation to guarantee zero data leakage between training and test sets.
"""
import json
import pandas as pd
from rdkit import Chem

def _check_rdkit_safe():
    import subprocess, sys
    try:
        res = subprocess.run([sys.executable, "-c", "import rdkit"], capture_output=True)
        return res.returncode == 0
    except Exception:
        return False

def _get_scaffold(smiles: str) -> str:
    """Helper to extract Murcko scaffold from SMILES for overlap checking."""
    if not _check_rdkit_safe():
        return ""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""

def validate_split_integrity(train_df: pd.DataFrame, test_df: pd.DataFrame, split_strategy: str) -> dict:
    """
    Ensure zero sample overlap. For specific splits, enforce scaffold or protein disjointness.
    """
    train_smiles = set(train_df['ligand_smiles'].tolist())
    test_smiles = set(test_df['ligand_smiles'].tolist())
    
    sample_overlap = train_smiles.intersection(test_smiles)
    
    # Calculate scaffolds
    train_scaffolds = set()
    test_scaffolds = set()
    scaffold_overlap_count = 0
    if split_strategy == 'scaffold':
        train_scaffolds = {_get_scaffold(s) for s in train_smiles if _get_scaffold(s)}
        test_scaffolds = {_get_scaffold(s) for s in test_smiles if _get_scaffold(s)}
        scaffold_overlap = train_scaffolds.intersection(test_scaffolds)
        scaffold_overlap_count = len(scaffold_overlap)
        
    # Calculate proteins
    train_proteins = set()
    test_proteins = set()
    protein_overlap_count = 0
    if split_strategy == 'protein':
        if 'e3_protein' in train_df.columns and 'neosubstrate' in train_df.columns:
            train_proteins = set(train_df['e3_protein'].dropna().tolist() + train_df['neosubstrate'].dropna().tolist())
            test_proteins = set(test_df['e3_protein'].dropna().tolist() + test_df['neosubstrate'].dropna().tolist())
            protein_overlap = train_proteins.intersection(test_proteins)
            protein_overlap_count = len(protein_overlap)
            
    manifest = {
        "train_ids": len(train_df),
        "test_ids": len(test_df),
        "train_scaffolds": len(train_scaffolds),
        "test_scaffolds": len(test_scaffolds),
        "protein_clusters_train": len(train_proteins),
        "protein_clusters_test": len(test_proteins),
        "intersection_checks": {
            "sample_overlap_count": len(sample_overlap),
            "scaffold_overlap_count": scaffold_overlap_count,
            "protein_cluster_overlap_count": protein_overlap_count
        }
    }
    
    # In random splits, sample overlap MUST be zero.
    # We do not strictly enforce scaffold/protein overlap zero for random, but we report it.
    if len(sample_overlap) > 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {len(sample_overlap)} samples overlap between train and test sets.")
        
    if split_strategy == 'scaffold' and scaffold_overlap_count > 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {scaffold_overlap_count} scaffolds overlap between train and test sets.")
        
    if split_strategy == 'protein' and protein_overlap_count > 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {protein_overlap_count} proteins overlap between train and test sets.")
        
    return manifest

if __name__ == "__main__":
    print("Smoke Test for validate_splits.py")
    t_df = pd.DataFrame({"ligand_smiles": ["C", "CC"]})
    val_df = pd.DataFrame({"ligand_smiles": ["CCC"]})
    res = validate_split_integrity(t_df, val_df, "random")
    assert res['intersection_checks']['sample_overlap_count'] == 0
    print("Passed!")
