"""
phase7_reproducibility_audit.py
================================
IFG-26 Phase 7 — Full Benchmark Reproducibility Audit

Recomputes benchmark metrics from raw data to ensure stability.
Runs 5 seeds × 5-fold CV.
Outputs:
    results/phase7/reproducibility_metrics.csv
    results/phase7/seed_variance_table.csv
    figures/reproducibility_errorbars.png
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
_MGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def setup_logging():
    lg = logging.getLogger("phase7_audit")
    lg.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    lg.addHandler(sh)
    return lg

def get_fp(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = _MGEN.GetFingerprint(mol)
            arr = np.zeros(2048, dtype=np.uint8)
            from rdkit.Chem import DataStructs
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
    except Exception: pass
    return None

def compute_metrics(y_true, y_score, k_frac=0.05):
    n = len(y_true)
    k = max(1, int(n * k_frac))
    order = np.argsort(y_score)[::-1]
    top_y = y_true[order[:k]]
    
    auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else np.nan
    auprc = average_precision_score(y_true, y_score) if len(np.unique(y_true))>1 else np.nan
    recall_k = np.sum(top_y) / np.sum(y_true) if np.sum(y_true) > 0 else 0
    
    prevalence = np.mean(y_true)
    precision_k = np.mean(top_y)
    lift_k = precision_k / prevalence if prevalence > 0 else 0
    brier = brier_score_loss(y_true, y_score)
    
    return auroc, auprc, recall_k, lift_k, brier

def load_data(lg):
    # For Phase 7 reproduction, we use PU Pool and PMD-v1
    pos_path = ROOT / "data/pu/pool_P_scaffold.parquet"
    pmd_path = ROOT / "dataset/phase2/test_scaffold.csv"
    
    if not pos_path.exists() or not pmd_path.exists():
        lg.error("Missing raw files")
        return None, None
        
    canon_path = ROOT / "dataset/phase1/canonicalized_compounds.csv"
    df_canon = pd.read_csv(canon_path, usecols=['inchi_key', 'canonical_smiles'])
    ik_map = dict(zip(df_canon['inchi_key'], df_canon['canonical_smiles']))
    
    pos_df = pd.read_parquet(pos_path)
    if 'smiles' not in pos_df.columns:
        pos_df['smiles'] = pos_df['ligand_inchikey'].map(ik_map)
    pos_smiles = pos_df['smiles'].dropna().tolist()[:1000] # Cap for speed during reproduction audit
    
    pmd_df = pd.read_csv(pmd_path)
    # test_scaffold.csv has no 'label' column — all rows are negatives (split=='test')
    if "split" in pmd_df.columns:
        pmd_df = pmd_df[pmd_df["split"] == "test"]
    # SMILES column name
    neg_smi_col = "canonical_smiles" if "canonical_smiles" in pmd_df.columns else "smiles"
    pmd_smiles = pmd_df[neg_smi_col].dropna().tolist()[:1000]
    
    pos_X = np.vstack([f for f in [get_fp(s) for s in pos_smiles] if f is not None])
    neg_X = np.vstack([f for f in [get_fp(s) for s in pmd_smiles] if f is not None])
    
    X = np.vstack([pos_X, neg_X])
    y = np.concatenate([np.ones(len(pos_X)), np.zeros(len(neg_X))])
    
    return X, y

def main():
    lg = setup_logging()
    lg.info("Phase 7 — Reproducibility Audit")
    
    res_dir = ROOT / "results/phase7"
    fig_dir = ROOT / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    X, y = load_data(lg)
    if X is None: return
    
    seeds = [42, 101, 1234, 777, 2026]
    all_results = []
    
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold = 0
        for train_idx, test_idx in skf.split(X, y):
            fold += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = LogisticRegression(max_iter=500, random_state=seed)
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, 1]
            
            auroc, auprc, recall, lift, brier = compute_metrics(y_test, y_score)
            
            all_results.append({
                "seed": seed, "fold": fold, 
                "auroc": auroc, "auprc": auprc,
                "recall@5": recall, "lift@5": lift, "brier": brier
            })
            
    df = pd.DataFrame(all_results)
    df.to_csv(res_dir / "reproducibility_metrics.csv", index=False)
    
    # Variance table
    summary = df.groupby("seed")[["auroc", "auprc", "recall@5", "lift@5"]].mean().reset_index()
    summary.to_csv(res_dir / "seed_variance_table.csv", index=False)
    lg.info(f"Mean AUROC across seeds: {summary['auroc'].mean():.4f} ± {summary['auroc'].std():.4f}")
    
    # Plot
    metrics = ["auroc", "auprc", "recall@5", "lift@5"]
    melted = df.melt(id_vars=["seed", "fold"], value_vars=metrics, var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="Metric", y="Score", capsize=.1, errorbar="sd", palette="viridis")
    plt.title("IFG-26 Reproducibility Audit (5 Seeds × 5 Folds)")
    plt.tight_layout()
    plt.savefig(fig_dir / "reproducibility_errorbars.png", dpi=300)
    plt.close()
    
    lg.info("Phase 7 Reproducibility Audit COMPLETE.")

if __name__ == "__main__":
    main()
