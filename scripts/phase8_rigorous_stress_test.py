"""
phase8_rigorous_stress_test.py
==============================
IFG-26 Phase 8 — Systematic Model Stress Test (Diagnostic Audit).
"""

import os
import sys
import json
import logging
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Environment Guards ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONNOUSERSITE"] = "1"

if sys.platform == "win32":
    env_base = r"C:\Users\Aadi Nair\miniconda3\miniconda4\envs\ifg26"
    torch_lib = os.path.join(env_base, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        if hasattr(os, "add_dll_directory"):
            try: os.add_dll_directory(torch_lib)
            except Exception: pass
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]

from rdkit import Chem

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT / "data" / "phase8_eval_bundles"
FEATS_DIR = ROOT / "data" / "features"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
DOCS_DIR = ROOT / "docs"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x): 
        return self.net(x).view(-1)

class SimpleMPNN(nn.Module):
    def __init__(self, node_in_dim=10, hidden_dim=64):
        super().__init__()
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        self.msg = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, nodes, adj):
        h = torch.relu(self.node_emb(nodes))
        m = torch.matmul(adj, h)
        h = torch.relu(self.msg(h + m))
        return h.mean(dim=1)

class InteractionModel(nn.Module):
    def __init__(self, prot_dim=1280, lig_dim=64):
        super().__init__()
        self.lig_enc = SimpleMPNN(hidden_dim=lig_dim)
        self.e3_enc = nn.Linear(prot_dim, 64)
        self.tgt_enc = nn.Linear(prot_dim, 64)
        self.trunk = MLP(lig_dim + 64 + 64 + 64 + 64, [128, 32])
    def forward(self, nodes, adj, e3, tgt):
        l_h = self.lig_enc(nodes, adj)
        e_h = torch.relu(self.e3_enc(e3))
        t_h = torch.relu(self.tgt_enc(tgt))
        le = l_h * e_h; lt = l_h * t_h
        return self.trunk(torch.cat([l_h, e_h, t_h, le, lt], dim=1))

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_ik_smiles():
    sm_map = {}
    for sub in ["train_scaffold.csv", "val_scaffold.csv", "test_scaffold.csv"]:
        p = ROOT / "dataset/phase2" / sub
        if p.exists():
            try:
                df = pd.read_csv(p, usecols=['compound_inchi_key', 'canonical_smiles'])
                sm_map.update(dict(zip(df['compound_inchi_key'], df['canonical_smiles'])))
            except Exception: pass
    return sm_map

def to_graph(smiles, max_nodes=40):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    nodes = np.zeros((max_nodes, 10), dtype=np.float32)
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    if not mol: return nodes, adj
    n = min(mol.GetNumAtoms(), max_nodes)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        nodes[i, 0] = atom.GetAtomicNumber()/100.0
        nodes[i, 1] = atom.GetDegree()/10.0
        nodes[i, 2] = float(atom.GetIsAromatic())
    for bond in mol.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i<max_nodes and j<max_nodes: adj[i,j] = adj[j,i] = 1.0
    return nodes, adj

def featurize(df, ecfp, sm_map, emb_map, prot_dim=1280):
    rows = df['ligand_feature_row'].clip(lower=0).values
    X_e = torch.from_numpy(ecfp[rows])
    n_list, a_list = [], []
    for ik in df['ligand_inchikey']:
        n, a = to_graph(sm_map.get(ik, ""))
        n_list.append(n); a_list.append(a)
    X_n, X_a = torch.from_numpy(np.stack(n_list)), torch.from_numpy(np.stack(a_list))
    def get_e(uid): return emb_map.get(uid, np.zeros(prot_dim, dtype=np.float32))
    X_e3 = torch.from_numpy(np.stack(df['e3_uniprot_id'].apply(get_e).values))
    X_t = torch.from_numpy(np.stack(df['target_uniprot_id'].apply(get_e).values))
    y = torch.from_numpy(df['label'].values.astype(np.float32)) if 'label' in df.columns else torch.zeros(len(df))
    return X_e, X_n, X_a, X_e3, X_t, y

def nnpu_loss_fn(p_out, u_out, pi=0.05):
    bce = nn.BCELoss()
    r_p_p = bce(p_out, torch.ones_like(p_out))
    r_p_n = bce(p_out, torch.zeros_like(p_out))
    r_u_n = bce(u_out, torch.zeros_like(u_out))
    return pi * r_p_p + torch.clamp(r_u_n - pi * r_p_n, min=0.0)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("Initializing...")
    ecfp = np.load(FEATS_DIR / "ligands_ecfp4.npy").astype(np.float32)
    sm_map = load_ik_smiles()
    prot_df = pd.read_parquet(FEATS_DIR / "protein_embeddings_esm2.parquet", engine='pyarrow')
    emb_map = {r['uniprot_id']: r['embedding'].astype(np.float32) for _, r in prot_df.iterrows() if r['fetch_ok']}
    
    p_pool = pd.read_parquet(ROOT / "data/pu/pool_P_scaffold.parquet", engine='pyarrow')
    u_pool = pd.read_parquet(ROOT / "data/pu/pool_U_scaffold.parquet", engine='pyarrow')
    trn_P = p_pool[p_pool['split'] == 'train']
    trn_U = u_pool.sample(min(len(u_pool), 2000))
    
    results = []
    regimes = ["S1_Proxy", "S3_Scaffold", "S6_PURanking"]

    for seed in [42, 101]:
        print(f"Seed {seed}")
        torch.manual_seed(seed); np.random.seed(seed)
        
        # M1 (LR baseline)
        m1 = LogisticRegression(max_iter=500); m1.fit(ecfp[trn_P['ligand_feature_row'].clip(0).values], trn_P['label'].values)
        
        # M2 (MLP)
        X_e, X_n, X_a, X_e3, X_t, y_t = featurize(trn_P, ecfp, sm_map, emb_map)
        m2_X = torch.cat([X_e, X_e3, X_t], dim=1)
        m2 = MLP(m2_X.shape[1]); opt2 = optim.Adam(m2.parameters(), lr=1e-3)
        for _ in range(2):
            for i in range(0, len(m2_X), 64):
                bx, by = m2_X[i:i+64], y_t[i:i+64]
                if len(bx)<2: continue
                opt2.zero_grad(); nn.BCELoss()(m2(bx), by).backward(); opt2.step()
        
        # M4 (nnPU)
        m4 = InteractionModel(); opt4 = optim.Adam(m4.parameters(), lr=1e-3)
        X_e_u, X_n_u, X_a_u, X_e3_u, X_t_u, _ = featurize(trn_U, ecfp, sm_map, emb_map)
        for _ in range(2):
            for i in range(0, len(X_n), 32):
                p_n, p_a, p_e3, p_t = X_n[i:i+32], X_a[i:i+32], X_e3[i:i+32], X_t[i:i+32]
                if len(p_n)<2: continue
                idx = np.random.randint(0, len(X_n_u), len(p_n))
                un_n, un_a, un_e3, un_t = X_n_u[idx], X_a_u[idx], X_e3_u[idx], X_t_u[idx]
                opt4.zero_grad(); nnpu_loss_fn(m4(p_n, p_a, p_e3, p_t), m4(un_n, un_a, un_e3, un_t)).backward(); opt4.step()

        # Eval
        for r in regimes:
            bp = BUNDLE_DIR / f"{r}.parquet"
            if not bp.exists(): continue
            df_b = pd.read_parquet(bp, engine='pyarrow')
            X_e_b, X_n_b, X_a_b, X_e3_b, X_t_b, y_b = featurize(df_b, ecfp, sm_map, emb_map)
            
            p1 = m1.predict_proba(ecfp[df_b['ligand_feature_row'].clip(0).values])[:, 1]
            auc1 = roc_auc_score(y_b, p1) if len(np.unique(y_b))>1 else 0.5
            
            m2.eval(); xj = torch.cat([X_e_b, X_e3_b, X_t_b], dim=1)
            with torch.no_grad(): p2 = m2(xj).numpy()
            auc2 = roc_auc_score(y_b, p2) if len(np.unique(y_b))>1 else 0.5
            
            m4.eval()
            with torch.no_grad(): p4 = m4(X_n_b, X_a_b, X_e3_b, X_t_b).numpy()
            auc4 = roc_auc_score(y_b, p4) if len(np.unique(y_b))>1 else 0.5
            
            results.append({"seed": seed, "regime": r, "model": "M1_LR", "auroc": auc1})
            results.append({"seed": seed, "regime": r, "model": "M2_MLP", "auroc": auc2})
            results.append({"seed": seed, "regime": r, "model": "M4_PU", "auroc": auc4})
            print(f"  {r}: M1={auc1:.2f} M4={auc4:.2f}")

    df_res = pd.DataFrame(results)
    pivot = df_res.groupby(['model', 'regime'])['auroc'].mean().unstack()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(RESULTS_DIR / "model_collapse_table.csv")
    plt.figure(figsize=(7, 4)); sns.heatmap(pivot, annot=True, cmap="YlGnBu"); plt.savefig(FIG_DIR / "model_collapse_heatmap.png")
    
    with open(DOCS_DIR / "stress_test_summary.md", "w") as f:
        f.write("# Stress Test Summary\n\n" + pivot.to_markdown() + "\n")
    print("Done.")

if __name__ == "__main__":
    main()
