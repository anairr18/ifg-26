"""
phase5G_data_scaling.py
========================
IFG-26 Phase 5G — Training Set Size Scaling (Learning Curves).

Evaluates how model performance (AUROC/Recall@5%) scales with 
available training data (10%, 25%, 50%, 100%).
Determines if the molecular glue task is data-limited or inherently complex.
"""

import os
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5G_scaling")
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5G_scaling.log", encoding="utf-8"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg

def get_fp(smi):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(str(smi))
    if m:
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return None

def main():
    lg = setup_logging()
    lg.info("Field Hardening: Starting Data Scaling Analysis...")

    # Load Full Dataset (Phase 2 Splits)
    curated_p2 = ROOT / "dataset/phase2"
    train_path = curated_p2 / "train_scaffold.csv"
    val_path = curated_p2 / "val_scaffold.csv"
    test_path = curated_p2 / "test_scaffold.csv"
    
    if not all(p.exists() for p in [train_path, val_path, test_path]):
        lg.error("Missing Phase 2 splits. Run Phase 2 first.")
        return
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    smi_col = "canonical_smiles"

    lg.info("Pre-calculating fingerprints...")
    X_train_full = np.stack([get_fp(s) for s in train_df[smi_col]])
    y_train_full = train_df["label"].values
    
    X_test = np.stack([get_fp(s) for s in test_df[smi_col]])
    y_test = test_df["label"].values

    fractions = [0.10, 0.25, 0.50, 1.00]
    results = []

    for frac in fractions:
        lg.info(f"Evaluating Scaling at {frac*100:.0f}%...")
        
        # Subsample Train
        n_samp = int(len(X_train_full) * frac)
        idx = np.random.choice(len(X_train_full), size=n_samp, replace=False)
        X_sub = X_train_full[idx]
        y_sub = y_train_full[idx]
        
        # Train & Eval
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        clf.fit(X_sub, y_sub)
        
        y_prob = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_prob)
        
        results.append({"Fraction": frac, "N_Train": n_samp, "AUROC": auroc})
        lg.info(f"  AUROC: {auroc:.3f}")

    res_df = pd.DataFrame(results)
    out_csv = ROOT / "data/diagnostics/data_scaling_results.csv"
    res_df.to_csv(out_csv, index=False)
    lg.info(f"Written: {out_csv.relative_to(ROOT)}")

    # Visualization
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 6))
    plt.plot(res_df["Fraction"] * 100, res_df["AUROC"], marker='o', color='darkorange', linewidth=2)
    plt.title("IFG-26 Learning Curve — Data Size Scaling", fontsize=14, fontweight='bold')
    plt.xlabel("Percentage of Training Data (%)", fontsize=12)
    plt.ylabel("Test AUROC (Scaffold Split)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "data_scaling_curve.png", dpi=300)
    lg.info(f"Saved: figures/data_scaling_curve.png")

    # Summary Report
    with open(ROOT / "docs/phase5G_scaling_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase 5G: Data Scaling Analysis\n\n")
        f.write("Evaluation of performance gains relative to training set volume.\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n## Scientific Analysis\n")
        
        slope = (res_df.iloc[-1]["AUROC"] - res_df.iloc[0]["AUROC"]) / (1.0 - 0.1)
        if slope > 0.1:
            f.write(f"📈 **Data Limited**: Performance is scaling strongly ({slope:.2f}). Adding more curated molecular glues will significantly improve models.\n")
        else:
            f.write(f"🛑 **Task Saturated**: Performance is plateauing ({slope:.2f}). Better architectures or more complex features are needed.\n")

    lg.info("Written: docs/phase5G_scaling_report.md")

if __name__ == "__main__":
    main()
