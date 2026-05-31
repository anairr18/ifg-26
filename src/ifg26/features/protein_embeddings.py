"""
protein_embeddings.py
=====================
Protein feature extraction using ESM2 models.
Supports batch processing, GPU acceleration, and result caching.
"""

import os
import torch
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False

class ESM2Embedder:
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = None):
        if not ESM_AVAILABLE:
            raise ImportError("fair-esm is not installed. Please install it to use ESM2Embedder.")
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading ESM2 model: {model_name} on {self.device}")
        
        # Determine representation layer based on model name
        if "t33" in model_name: self.repr_layer = 33
        elif "t30" in model_name: self.repr_layer = 30
        elif "t12" in model_name: self.repr_layer = 12
        elif "t6" in model_name: self.repr_layer = 6
        else: self.repr_layer = 33 # Default for 650M
        
        self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        self.model = self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.embed_dim = self.model.embed_dim
        
    def embed_sequences(self, sequences: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Extract mean-pooled embeddings for a list of sequences.
        Truncates sequences longer than 1022 residues.
        """
        all_embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            # Handle truncation and formatting for batch converter
            # (label, seq)
            formatted_batch = [(f"seq_{j}", s[:1022]) for j, s in enumerate(batch_seqs)]
            
            labels, strs, tokens = self.batch_converter(formatted_batch)
            tokens = tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
                token_representations = results["representations"][self.repr_layer]
                
            # Mean pooling over sequence length (excluding BOS/EOS)
            for j, (_, seq) in enumerate(formatted_batch):
                # tokens[j] has BOS at 0, residues 1:len+1, EOS at len+1
                seq_len = len(seq)
                mean_vec = token_representations[j, 1:seq_len + 1].mean(0).cpu().numpy()
                all_embeddings.append(mean_vec)
                
        return np.array(all_embeddings, dtype=np.float32)

def compute_sequence_hash(seq: str) -> str:
    """Deterministic hash of sequence for cache keys."""
    return hashlib.sha256(seq.strip().upper().encode()).hexdigest()

def get_cached_embeddings(uniprot_ids: List[str], sequences: List[str], 
                          cache_path: Path, model_name: str = "esm2_t33_650M_UR50D") -> pd.DataFrame:
    """
    Retrieves or computes ESM2 embeddings with caching.
    """
    if cache_path.exists():
        cache_df = pd.read_parquet(cache_path)
    else:
        cache_df = pd.DataFrame(columns=["uniprot_id", "hash", "embedding"])

    # Create mapping for current request
    req_hashes = [compute_sequence_hash(s) for s in sequences]
    req_df = pd.DataFrame({
        "uniprot_id": uniprot_ids,
        "sequence": sequences,
        "hash": req_hashes
    })
    
    # Identify what needs computation
    merged = req_df.merge(cache_df[["hash", "embedding"]], on="hash", how="left")
    missing_mask = merged["embedding"].isna()
    
    if missing_mask.any():
        missing_seqs = merged.loc[missing_mask, "sequence"].unique().tolist()
        print(f"Computing ESM2 embeddings for {len(missing_seqs)} unique sequences...")
        
        embedder = ESM2Embedder(model_name=model_name)
        new_embeddings = embedder.embed_sequences(missing_seqs)
        
        # Update cache
        new_data = []
        for seq, emb in zip(missing_seqs, new_embeddings):
            h = compute_sequence_hash(seq)
            new_data.append({"hash": h, "embedding": emb.tolist()})
        
        new_cache_df = pd.concat([cache_df, pd.DataFrame(new_data)], ignore_index=True)
        # Handle duplicates if any hashes collided or were re-added
        new_cache_df = new_cache_df.drop_duplicates(subset=["hash"])
        new_cache_df.to_parquet(cache_path, index=False)
        
        # Re-merge to fill missing
        merged = req_df.merge(new_cache_df[["hash", "embedding"]], on="hash", how="left")
        
    return merged
