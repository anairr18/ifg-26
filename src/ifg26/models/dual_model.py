"""
dual_model.py
=============
Dual-modality neural network model for IFG-26.
Combines ECFP4 ligand fingerprints and ESM2 protein embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional

class DualModalityMLP(nn.Module):
    def __init__(self, 
                 ligand_dim: int = 2048, 
                 protein_dim: int = 1280, 
                 n_proteins: int = 2, # E3 + Target
                 ligand_hidden: list[int] = [512, 256],
                 protein_hidden: list[int] = [512, 256],
                 fusion_hidden: list[int] = [512, 128],
                 dropout: float = 0.3,
                 batch_norm: bool = True):
        super().__init__()
        
        # 1. Ligand Encoder
        self.ligand_encoder = self._make_mlp(ligand_dim, ligand_hidden, dropout, batch_norm)
        
        # 2. Protein Encoder (Combined E3 + Target)
        self.protein_encoder = self._make_mlp(protein_dim * n_proteins, protein_hidden, dropout, batch_norm)
        
        # 3. Fusion Head
        # If ligand_hidden and protein_hidden are non-empty, the input to fusion is the sum of their last hidden dims
        in_fusion = ligand_hidden[-1] + protein_hidden[-1]
        self.fusion_head = self._make_mlp(in_fusion, fusion_hidden, dropout, batch_norm)
        
        # 4. Final Output Linear Layer
        self.output_layer = nn.Linear(fusion_hidden[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def _make_mlp(self, in_dim, hidden_dims, dropout, batch_norm):
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        return nn.Sequential(*layers)

    def forward(self, x_ligand: torch.Tensor, x_protein: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Supports both split inputs and single concatenated input.
        """
        if x_protein is None:
            # Assume concatenated: (L, P) where L=2048, P=dimension of ESM vecs
            x_protein = x_ligand[:, 2048:]
            x_ligand = x_ligand[:, :2048]
            
        h_lig = self.ligand_encoder(x_ligand)
        h_prot = self.protein_encoder(x_protein)
        fused = torch.cat([h_lig, h_prot], dim=1)
        out = self.fusion_head(fused)
        logits = self.output_layer(out)
        return self.sigmoid(logits).squeeze(-1)

    def score_logit(self, x_ligand: torch.Tensor, x_protein: torch.Tensor) -> torch.Tensor:
        """Return scalar score BEFORE sigmoid for nnPU losses if needed."""
        h_lig = self.ligand_encoder(x_ligand)
        h_prot = self.protein_encoder(x_protein)
        fused = torch.cat([h_lig, h_prot], dim=1)
        out = self.fusion_head(fused)
        return self.output_layer(out).squeeze(-1)
