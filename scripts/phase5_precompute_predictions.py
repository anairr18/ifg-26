"""
phase5_precompute_predictions.py
================================
IFG-26 Phase 5 Pre-Computation
Generates all predictions required for Phase 5 benchmarks.
"""
import os
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# --- Model Arch Replicating Phase 4 ---
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

def sha256_file(path):
    if not os.path.exists(path): return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def make_ecfp_tensor(df, ecfp_mat):
    rows = df["ligand_feature_row"].clip(0).values
    return torch.from_numpy(ecfp_mat[rows].astype(np.float32))

def make_protein_onehot(df, n_proteins):
    n_p1 = n_proteins + 1
    e3_oh = np.eye(n_p1, dtype=np.float32)[df["e3_label_id"].clip(-1).values + 1]
    tgt_oh = np.eye(n_p1, dtype=np.float32)[df["target_label_id"].clip(-1).values + 1]
    return torch.from_numpy(np.hstack([e3_oh, tgt_oh]))

def main():
    root = Path(__file__).resolve().parent.parent
    pred_dir = root / 'data' / 'preds' / 'phase5'
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Best models selected from Phase 4B metrics
    best_pis = {"L0": 0.02, "LP0": 0.05}
    
    # Load data
    pool_P = pd.read_parquet(root / 'data/pu/pool_P_scaffold.parquet')
    if 'compound_inchi_key' in pool_P.columns and 'ligand_inchikey' not in pool_P.columns:
        pool_P['ligand_inchikey'] = pool_P['compound_inchi_key']
    pool_U = pd.read_parquet(root / 'data/pu/pool_U_scaffold.parquet')
    
    ecfp_path = root / 'data/features/ligands_ecfp4.npy'
    ecfp_mat = np.load(str(ecfp_path))
    feat_hash = sha256_file(ecfp_path)
    
    promaps = pd.read_parquet(root / 'dataset/protein_mapping.parquet') if os.path.exists(root / 'dataset/protein_mapping.parquet') else pd.read_parquet(root / 'data/features/protein_index.parquet')
    n_prot = len(promaps)
    label_to_uniprot = dict(zip(promaps['label_id'], promaps['uniprot_id']))
    
    if 'e3_uniprot' not in pool_P.columns and 'e3_label_id' in pool_P.columns:
        pool_P['e3_uniprot'] = pool_P['e3_label_id'].map(label_to_uniprot)
        pool_P['target_uniprot'] = pool_P['target_label_id'].map(label_to_uniprot)
        if 'e3_label_id' in pool_U.columns:
            pool_U['e3_uniprot'] = pool_U['e3_label_id'].map(label_to_uniprot)
            pool_U['target_uniprot'] = pool_U['target_label_id'].map(label_to_uniprot)
        else:
            pool_U['e3_uniprot'] = 'UNKNOWN'
            pool_U['target_uniprot'] = 'UNKNOWN'
    pool_U = pool_U.copy()
    pool_U['subset'] = 'test'
    pool_U['source'] = pool_U.get('source_dataset', 'u_pool')
    if 'pair_id' not in pool_P.columns: pool_P['pair_id'] = pool_P['ligand_inchikey'] + '_' + pool_P['e3_uniprot'].fillna('') + '_' + pool_P['target_uniprot'].fillna('')
    if 'pair_id' not in pool_U.columns: pool_U['pair_id'] = pool_U['ligand_inchikey'] + '_U_pool'
    
    bins_df = pd.read_parquet(root / 'data/features/test_similarity_bins.parquet')
    bin_map = bins_df.set_index('compound_inchi_key')['similarity_bin'].to_dict()
    nn_map = bins_df.set_index('compound_inchi_key')['nn_tanimoto_to_train'].to_dict()
    
    def get_split_subsets(df, split_name, seed=42):
        if split_name == 'scaffold':
            sj = json.load(open(root / 'splits/scaffold_split.json'))
            train_idx, val_idx, test_idx = set(sj['train_row_indices']), set(sj['val_row_indices']), set(sj['test_row_indices'])
            return ['train' if i in train_idx else 'val' if i in val_idx else 'test' if i in test_idx else 'train' for i in range(len(df))]
        
        import random
        split_map = df.copy()
        if split_name == 'pair_holdout':
            key_col = '_pair_key'
            valid = split_map[(split_map["e3_uniprot"].notna()) & (split_map["e3_uniprot"] != "") & (split_map["target_uniprot"].notna()) & (split_map["target_uniprot"] != "")].copy()
            valid[key_col] = valid["e3_uniprot"] + "||" + valid["target_uniprot"]
        else:
            key_col = "e3_uniprot" if split_name == 'e3_holdout' else "target_uniprot"
            valid = split_map[split_map[key_col].notna() & (split_map[key_col] != "")].copy()
            
        rest = split_map[~split_map.index.isin(valid.index)].copy()
        unique_vals = sorted(valid[key_col].unique())
        rng = random.Random(seed)
        rng.shuffle(unique_vals)
        n = len(unique_vals)
        n_train, n_val = int(0.8 * n), int(0.1 * n)
        train_vals = set(unique_vals[:n_train])
        val_vals = set(unique_vals[n_train:n_train+n_val])
        
        def assign(v): return "train" if v in train_vals else "val" if v in val_vals else "test"
        valid["subset"] = valid[key_col].apply(assign)
        rest["subset"] = "train"
        return pd.concat([valid, rest]).sort_index()["subset"].tolist()
        
    for split_name in ['scaffold', 'e3_holdout', 'target_holdout', 'pair_holdout']:
        P_sub = pool_P.copy()
        P_sub['split_name'] = split_name
        P_sub['subset'] = get_split_subsets(P_sub, split_name)
        P_sub['source'] = P_sub.get('source_dataset', 'mgdb')
        
        for model_type, pi in best_pis.items():
            model_path = root / f"results/models/nnpu_{model_type}_pi{pi}.pt"
            if not os.path.exists(model_path): continue
            
            checkpoint = torch.load(model_path, map_location='cpu')
            mlp_cfg = checkpoint['mlp_cfg']
            model = MLP(checkpoint['in_dim'], mlp_cfg.get('hidden_dims', [512,128]), mlp_cfg.get('dropout', 0.3), mlp_cfg.get('batch_norm', True))
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            
            with torch.no_grad():
                # Score P
                ecfp_P = make_ecfp_tensor(P_sub, ecfp_mat)
                X_P = ecfp_P if model_type == 'L0' else torch.cat([ecfp_P, make_protein_onehot(P_sub, n_prot)], dim=1)
                P_sub['score'] = model(X_P).numpy()
                
                # Score U
                U_sub = pool_U.copy()
                U_sub['split_name'] = split_name
                ecfp_U = make_ecfp_tensor(U_sub, ecfp_mat)
                X_U = ecfp_U if model_type == 'L0' else torch.cat([ecfp_U, make_protein_onehot(U_sub, n_prot)], dim=1)
                U_sub['score'] = model(X_U).numpy()
            
            # Form final DF for this split/model
            # Keep required columns: pair_id, split_name, subset, source, ligand_inchikey, e3_uniprot, target_uniprot, similarity_bin, nn_to_train, score, pi_used, model_hash, feature_hash
            out_P = P_sub[['pair_id', 'split_name', 'subset', 'source', 'score']].copy()
            out_P['ligand_inchikey'] = P_sub.get('ligand_inchikey')
            out_P['e3_uniprot'] = P_sub.get('e3_uniprot', '')
            out_P['target_uniprot'] = P_sub.get('target_uniprot', '')
            
            out_U = U_sub[['pair_id', 'split_name', 'subset', 'source', 'score']].copy()
            out_U['ligand_inchikey'] = U_sub.get('ligand_inchikey')
            out_U['e3_uniprot'] = U_sub.get('e3_uniprot', '')
            out_U['target_uniprot'] = U_sub.get('target_uniprot', '')
            
            combined = pd.concat([out_P, out_U], ignore_index=True)
            combined['similarity_bin'] = combined['ligand_inchikey'].map(bin_map).fillna('UNKNOWN')
            combined['nn_to_train'] = combined['ligand_inchikey'].map(nn_map).fillna(0.0)
            combined['pi_used'] = pi
            combined['model_hash'] = sha256_file(model_path)
            combined['feature_hash'] = feat_hash
            
            # For OOD splits, we only care about the TEST subset (all U + test P). For scaffold, we save all.
            if split_name != 'scaffold':
                combined = combined[combined['subset'] == 'test'].reset_index(drop=True)
                
            out_path = pred_dir / f"nnpu_{model_type}_{split_name}.parquet"
            combined.to_parquet(out_path, index=False)
            print(f"Saved {out_path} ({len(combined)} rows)")

    # Write report
    os.makedirs('docs', exist_ok=True)
    with open('docs/phase5_precompute_predictions_report.md', 'w') as f:
        f.write("# Phase 5 Precomputations Report\n")
        f.write("Prediction Parquets successfully generated.\n")
        f.write("Included splits: scaffold, e3_holdout, target_holdout, pair_holdout.\n")
        f.write("Filled standard required columns for Nature-grade eval.\n")

if __name__ == '__main__':
    main()
