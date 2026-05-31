"""
hard_negatives.py
=================
Implementation of fine-grained hard negatives (Tanimoto-near neighbors and MMPs).
Focuses on local indistinguishability in chemical feature space.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from typing import List, Tuple, Optional

def calculate_properties(smiles: str) -> Tuple[float, float]:
    """Calculate MW and LogP for a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 0.0, 0.0
    return Descriptors.MolWt(mol), Descriptors.MolLogP(mol)

def find_tanimoto_near_neighbors(query_smiles: str, 
                                 universe_df: pd.DataFrame, 
                                 universe_fps: List,
                                 threshold_min: float = 0.85, 
                                 threshold_max: float = 0.95,
                                 mw_tol: float = 50.0,
                                 logp_tol: float = 1.0,
                                 n_max: int = 5) -> List[str]:
    """
    Finds compounds in universe_df that are similar to query_smiles 
    within a Tanimoto range and match MW/LogP properties.
    """
    q_mol = Chem.MolFromSmiles(query_smiles)
    if q_mol is None: return []
    q_fp = AllChem.GetMorganFingerprintAsBitVect(q_mol, 2, nBits=2048)
    q_mw, q_logp = calculate_properties(query_smiles)

    # 1. Broad property filter
    mask = (universe_df['mw'] >= q_mw - mw_tol) & (universe_df['mw'] <= q_mw + mw_tol) & \
           (universe_df['logp'] >= q_logp - logp_tol) & (universe_df['logp'] <= q_logp + logp_tol)
    
    indices = np.where(mask)[0]
    
    results = []
    for idx in indices:
        c_fp = universe_fps[idx]
        if c_fp is None: continue
        
        sim = DataStructs.TanimotoSimilarity(q_fp, c_fp)
        if threshold_min <= sim <= threshold_max:
            results.append((universe_df.iloc[idx]['smiles'], sim))
            
    results.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in results[:n_max]]

def generate_local_perturbations(smiles: str, n_mutations: int = 10) -> List[str]:
    """
    Generates Matched Molecular Pairs (MMP)-style perturbations by 
    swapping atoms or small groups.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return []
    
    replacements = {6: [7, 8], 17: [9, 35], 7: [6, 8]}
    results = set()
    
    atoms = list(mol.GetAtoms())
    np.random.shuffle(atoms)
    
    for atom in atoms:
        idx = atom.GetIdx()
        symbol_idx = atom.GetAtomicNum()
        if symbol_idx in replacements:
            for new_sym in replacements[symbol_idx]:
                rw_mol = Chem.RWMol(mol)
                rw_mol.ReplaceAtom(idx, Chem.Atom(new_sym))
                try:
                    res_smiles = Chem.MolToSmiles(rw_mol)
                    if Chem.MolFromSmiles(res_smiles):
                        results.add(res_smiles)
                except:
                    continue
        if len(results) >= n_mutations: break
                
    return list(results)

def build_hard_near_tier(positives_df: pd.DataFrame, 
                         universe_df: pd.DataFrame, 
                         n_per_pos: int = 10) -> pd.DataFrame:
    """
    Main entry point to build the HARD-NEAR negative tier.
    """
    all_negatives = []
    
    print("Pre-calculating universe fingerprints...")
    universe_fps = []
    for smi in universe_df['smiles']:
        m = Chem.MolFromSmiles(smi)
        if m:
            universe_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
        else:
            universe_fps.append(None)
    
    print(f"Processing {len(positives_df)} positives...")
    for _, row in positives_df.iterrows():
        p_smiles = row['ligand_smiles']
        
        near_negatives = find_tanimoto_near_neighbors(p_smiles, universe_df, universe_fps)
        mutant_negatives = generate_local_perturbations(p_smiles)
        
        combined = list(set(near_negatives + mutant_negatives))
        
        for neg in combined:
            all_negatives.append({
                'ligand_smiles': neg,
                'parent_positive': p_smiles,
                'source_type': 'hard_near',
                'e3_uniprot_id': row.get('e3_uniprot_id'),
                'target_uniprot_id': row.get('target_uniprot_id'),
                'label': 0 # Weak negative/unlabeled
            })
            if len(all_negatives) >= len(positives_df) * n_per_pos: break
            
    return pd.DataFrame(all_negatives)
