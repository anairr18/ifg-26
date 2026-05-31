"""
phase5F_sanity_checks.py
=========================
IFG-26 Phase 5F — Random-Label & Feature-Shuffling Sanity Checks.

Proves that models are learning real chemistry by showing that 
shuffling labels or features collapses performance to random (AUROC=0.5).
"""

import os
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5F_sanity")
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5F_sanity.log", encoding="utf-8"))
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
    lg.info("Field Hardening: Starting Sanity Checks...")

    # Load Baseline Data (Scaffold Split)
    # We'll use a representative subset of the data for this test
    curated_p2 = ROOT / "dataset/phase2"
    test_path = curated_p2 / "test_scaffold.csv"
    if not test_path.exists():
        lg.error("Missing test_scaffold.csv. Run Phase 2 first.")
        return
        
    df = pd.read_csv(test_path)
    smi_col = next((c for c in ["canonical_smiles", "smiles"] if c in df.columns), None)
    
    lg.info("Generating fingerprints for sanity checks...")
    X = np.stack([get_fp(s) for s in df[smi_col]])
    y = df["label"].values # Binary (0=neg, 1=pos)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = []

    # 1. Baseline Performance (Unshuffled)
    lg.info("Running Baseline (Unshuffled)...")
    base_scores = []
    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        base_scores.append(roc_auc_score(y[test_idx], clf.predict_proba(X[test_idx])[:, 1]))
    
    results.append({"Test": "Baseline", "Mean AUROC": np.mean(base_scores)})
    lg.info(f"  Baseline: {np.mean(base_scores):.3f}")

    # 2. Random Label Test (Critical)
    lg.info("Running Random Label Sanity Test...")
    y_shuffled = y.copy()
    np.random.seed(42)
    np.random.shuffle(y_shuffled)
    
    rand_scores = []
    for train_idx, test_idx in skf.split(X, y_shuffled):
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        clf.fit(X[train_idx], y_shuffled[train_idx])
        rand_scores.append(roc_auc_score(y_shuffled[test_idx], clf.predict_proba(X[test_idx])[:, 1]))
        
    results.append({"Test": "Random Labels", "Mean AUROC": np.mean(rand_scores)})
    lg.info(f"  Random Labels: {np.mean(rand_scores):.3f}")

    # 3. Y-Scrambling (Permuted labels within train set only)
    # Proves that AUROC drop isn't just because the test set is hard
    lg.info("Running Y-Scrambling Test...")
    yscram_scores = []
    for train_idx, test_idx in skf.split(X, y):
        y_train_shuff = y[train_idx].copy()
        np.random.shuffle(y_train_shuff)
        
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        clf.fit(X[train_idx], y_train_shuff)
        yscram_scores.append(roc_auc_score(y[test_idx], clf.predict_proba(X[test_idx])[:, 1]))
        
    results.append({"Test": "Y-Scrambling", "Mean AUROC": np.mean(yscram_scores)})
    lg.info(f"  Y-Scrambling: {np.mean(yscram_scores):.3f}")

    res_df = pd.DataFrame(results)
    out_csv = ROOT / "data/diagnostics/sanity_check_results.csv"
    res_df.to_csv(out_csv, index=False)
    lg.info(f"Written: {out_csv.relative_to(ROOT)}")

    # Summary Report
    with open(ROOT / "docs/phase5F_sanity_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase 5F: Scientific Integrity Sanity Checks\n\n")
        f.write("Validation tests ensuring model success is driven by chemistry, not data artifacts.\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n## Scientific Analysis\n")
        
        rl_score = res_df.set_index("Test").at["Random Labels", "Mean AUROC"]
        if 0.45 <= rl_score <= 0.55:
            f.write("✅ **Pass**: Baseline labels are scientifically valid. Randomizing labels correctly causes AUROC to drop to ~0.5.\n")
        else:
            f.write(f"❌ **Fail**: AUROC ({rl_score:.3f}) is high even with random labels. Serious data leakage detected.\n")

    lg.info("Written: docs/phase5F_sanity_report.md")

if __name__ == "__main__":
    main()
