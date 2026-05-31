"""
scorer.py
=========
Production-ready scoring interface for IFG-26.
Optimized for high-throughput virtual screening.
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Union, Optional, Dict

from ifg26.features.protein_embeddings import ESM2Embedder
from ifg26.models.dual_model import DualModalityMLP

class IFG26Scorer:
    """
    High-throughput scoring engine for IFG-26 molecular glue prediction.
    """
    def __init__(self, 
                 model_path: str, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 esm_model_name: str = "esm2_t33_650M_UR50D"):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        
        # 1. Initialize Embedder FIRST to get dimension
        self.embedder = ESM2Embedder(model_name=esm_model_name, device=device)
        self.embed_dim = self.embedder.model.embed_tokens.embedding_dim
        
        # 2. Load Model using detected dimension
        self.model = self._load_model(model_path, self.embed_dim)
        self.model.eval()
        self.model.to(self.device)
        
        # 3. Cache for protein embeddings
        self.protein_cache = {}

    def _load_model(self, path: Path, protein_dim: int) -> DualModalityMLP:
        """Loads the trained model architecture and weights."""
        model = DualModalityMLP(ligand_dim=2048, protein_dim=protein_dim, n_proteins=2)
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    def get_ligand_features(self, smiles_list: List[str]) -> torch.Tensor:
        """Vectorized ECFP4 generation."""
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros((1,))
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            else:
                fps.append(np.zeros(2048))
        return torch.tensor(np.array(fps), dtype=torch.float32).to(self.device)

    def get_protein_features(self, sequences: List[str]) -> torch.Tensor:
        """Efficient ESM2 embedding with internal caching."""
        embeddings = []
        new_seqs = []
        
        for seq in sequences:
            if seq in self.protein_cache:
                embeddings.append(self.protein_cache[seq])
            else:
                new_seqs.append(seq)
                embeddings.append(None) # Placeholder
                
        if new_seqs:
            new_embs = self.embedder.embed_sequences(new_seqs)
            idx = 0
            for i, emb in enumerate(embeddings):
                if emb is None:
                    e = new_embs[idx]
                    embeddings[i] = e
                    self.protein_cache[sequences[i]] = e
                    idx += 1
                    
        return torch.tensor(np.array(embeddings), dtype=torch.float32).to(self.device)

    def score(self, 
              smiles: Union[str, List[str]], 
              e3_seq: str, 
              target_seq: str, 
              batch_size: int = 128) -> np.ndarray:
        """
        Main scoring function. Splits inputs into batches for throughput.
        """
        if isinstance(smiles, str):
            smiles = [smiles]
            
        n = len(smiles)
        all_scores = []
        
        # Pre-embed proteins (reused for the entire ligand batch)
        e3_emb = self.get_protein_features([e3_seq])
        target_emb = self.get_protein_features([target_seq])
        p_feat = torch.cat([e3_emb, target_emb], dim=1) # (1, 2560)

        for i in range(0, n, batch_size):
            batch_smiles = smiles[i:i+batch_size]
            l_feat = self.get_ligand_features(batch_smiles)
            
            # Expand p_feat to match batch size
            p_feat_batch = p_feat.expand(len(l_feat), -1)
            
            with torch.no_grad():
                scores = self.model(l_feat, p_feat_batch)
                all_scores.append(scores.cpu().numpy())
                
        return np.concatenate(all_scores)

def virtual_flow_example():
    """Example integration script for high-throughput screening."""
    # 1. Initialize Scorer
    # scorer = IFG26Scorer("path/to/trained_model.pt")
    
    # 2. Define Proteins
    e3_seq = "MKTVRQERLKSIVRILERS"
    target_seq = "MKTVRQERLKSIVRILERS"
    
    # 3. Large SMILES set (simulated)
    smiles_list = ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"] * 1000
    
    print(f"Scoring {len(smiles_list)} pairs...")
    # results = scorer.score(smiles_list, e3_seq, target_seq)
    # print(f"Top Score: {np.max(results)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print("CLI Tool Entry Point (To be implemented with argparse if needed)")
    else:
        virtual_flow_example()
