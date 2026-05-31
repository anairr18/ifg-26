"""
generate_splits.py

Logic for generating different benchmark splits (random, scaffold, protein holdout).
"""
import pandas as pd

def generate_random_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Standard random split as the baseline benchmark."""
    try:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df
    except Exception as e:
        # Fallback manual split if sklearn fails natively
        idx = int(len(df) * (1 - test_size))
        df_shuf = df.sample(frac=1, random_state=random_state)
        return df_shuf.iloc[:idx], df_shuf.iloc[idx:]

def generate_scaffold_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Bemis-Murcko scaffold holdout. 
    Groups ligands by their Murcko Scaffold and bins the groups greedily
    to enforce disjoint scaffold partitions between Train and Test sets.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ImportError:
        print("WARNING: RDKit not installed. Falling back to random split.")
        return generate_random_split(df, test_size=test_size, random_state=random_state)
        
    smiles_col = None
    for col in ['ligand_smiles', 'compound_smiles', 'smiles']:
        if col in df.columns:
            smiles_col = col
            break
            
    if smiles_col is None:
        print("WARNING: No ligand SMILES column found. Falling back to random split.")
        return generate_random_split(df, test_size=test_size, random_state=random_state)
        
    def get_scaffold(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            return MurckoScaffoldSmiles(mol=mol)
        except Exception:
            return ""
            
    df = df.copy()
    df['_scaffold'] = df[smiles_col].apply(get_scaffold)
    
    # Group indices by scaffold
    scaffold_groups = df.groupby('_scaffold').apply(lambda g: g.index.tolist()).tolist()
    
    import numpy as np
    rng = np.random.RandomState(random_state)
    rng.shuffle(scaffold_groups)
    
    train_indices = []
    test_indices = []
    target_test_count = len(df) * test_size
    
    for group in scaffold_groups:
        if len(test_indices) < target_test_count:
            test_indices.extend(group)
        else:
            train_indices.extend(group)
            
    train_df = df.loc[train_indices].drop(columns=['_scaffold'])
    test_df = df.loc[test_indices].drop(columns=['_scaffold'])
    return train_df, test_df

def generate_protein_holdout_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Sequence identity / target holdout.
    Groups protein-ligand pairs by the protein's UniProt ID and bins the groups
    greedily to guarantee that Train and Test sets have sequence-disjoint targets.
    """
    protein_col = None
    for col in ['target_uniprot_id', 'e3_uniprot_id', 'uniprot_id', 'protein_id']:
        if col in df.columns:
            protein_col = col
            break
            
    if protein_col is None:
        print("WARNING: No UniProt ID column found. Falling back to random split.")
        return generate_random_split(df, test_size=test_size, random_state=random_state)
        
    df = df.copy()
    protein_groups = df.groupby(protein_col).apply(lambda g: g.index.tolist()).tolist()
    
    import numpy as np
    rng = np.random.RandomState(random_state)
    rng.shuffle(protein_groups)
    
    train_indices = []
    test_indices = []
    target_test_count = len(df) * test_size
    
    for group in protein_groups:
        if len(test_indices) < target_test_count:
            test_indices.extend(group)
        else:
            train_indices.extend(group)
            
    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]
    return train_df, test_df

def get_split_func(split_name: str):
    """Factory to get the right split function"""
    if split_name == 'random':
        return generate_random_split
    elif split_name == 'scaffold':
        return generate_scaffold_split
    elif split_name == 'protein':
        return generate_protein_holdout_split
    else:
        raise ValueError(f"Unknown split type: {split_name}")

if __name__ == "__main__":
    print("Smoke Test for generate_splits.py")
    df = pd.DataFrame({"ligand_smiles": ["C", "CC", "CCC", "CCCC"], "label": [0, 1, 0, 1]})
    tr, te = generate_random_split(df, test_size=0.5)
    assert len(tr) == 2 and len(te) == 2
    print("Passed!")
