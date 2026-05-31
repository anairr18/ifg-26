"""
phase8_adversarial_optimization.py
==================================

Case Study: Cereblon-Ikaros Molecular Glue Optimization Loop.
Generates 'intelligent negatives' (adversarial 0s) to audit model resolution.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

# Model Architecture
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float, batch_norm: bool):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

ROOT = Path(__file__).resolve().parent.parent

def localized_mutate(smiles):
    """Localized chemical mutations (atom atom swaps, deletions)."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return []
    # 1. Swapping Oxygen for Carbon (simulating functional group change)
    mutants = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == 'O':
            nm = Chem.RWMol(mol)
            nm.GetAtomWithIdx(i).SetAtomicNum(6) # Change O -> C (Adversarial)
            try:
                Chem.SanitizeMol(nm)
                mutants.append(Chem.MolToSmiles(nm))
            except: pass
    return list(set(mutants))

def main():
    print("Running Adversarial Optimization: Cereblon/Ikaros Case Study...")

    # Load Model (using previous seed 0.05 checkpoint)
    ckpt_path = ROOT / "results/models/nnpu_LP0_pi0.05.pt"
    if not ckpt_path.exists():
        print("Model M2 (nnPU-LP0) not found.")
        return
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = MLP(ckpt['in_dim'], ckpt['mlp_cfg']['hidden_dims'], 
                ckpt['mlp_cfg']['dropout'], ckpt['mlp_cfg']['batch_norm'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Known Glue: Thalidomide scaffold
    seed_smi = "O=C1CCc(N1C(=O)C2=CC=CC=C2)C=O" # Simplified Thalidomide proxy
    
    # 1. Generative Step: Mutate Ligand
    mutants = localized_mutate(seed_smi)
    print(f"Generated {len(mutants)} intelligent decoys.")
    
    # Evaluate Loop
    results = []
    e3_prot = "Q13422"; target_prot = "Q13421" # Mock IDs
    
    with torch.no_grad():
        for mut_smi in mutants + [seed_smi]:
            mol = Chem.MolFromSmiles(mut_smi)
            if not mol: continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros(2048, dtype=np.float32)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            
            # Combine ECFP (2048) + Mock Protein OH (428)
            feat = torch.zeros((1, ckpt['in_dim']))
            feat[0, :2048] = torch.from_numpy(arr)
            # (In a real case, set protein bits 2048:2048+428)
            
            score = model(feat).item()
            results.append({'smiles': mut_smi, 'score': score, 'type': 'mutant' if mut_smi != seed_smi else 'seed'})

    res_df = pd.DataFrame(results).sort_values('score', ascending=False)
    print("\nResolution Audit: Seed vs Intelligent Decoys")
    print(res_df)
    
    # Benchmark Flag
    if (res_df['score'].max() - res_df['score'].min()) < 0.1:
        print("\n[WARNING] Model resolution failure: Seed and near-neighbor decoy scores are indistinguishable.")

if __name__ == "__main__":
    main()
