"""
phase7_calibration_analysis.py
==============================
Phase 7 - Calibration and Reliability Analysis
Computes ECE, Brier score, and creates reliability diagrams.
Outputs:
    figures/reliability_diagram.png
    figures/calibration_comparison.png
"""

import os, sys, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

def setup_log():
    lg = logging.getLogger("phase7_cal")
    lg.setLevel(logging.INFO)
    lg.addHandler(logging.StreamHandler(sys.stdout))
    return lg

def main():
    lg = setup_log()
    lg.info("Phase 7 — Calibration Analysis (Real Out-of-Sample Evaluation)")
    
    fig_dir = ROOT / "figures"
    feats_dir = ROOT / "data" / "features"
    pu_dir = ROOT / "data" / "pu"
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load Features and Indices ─────────────────────────────────────
    lg.info("Loading features and splits...")
    ecfp = np.load(feats_dir / "ligands_ecfp4.npy").astype(np.float32)
    lig_idx = pd.read_parquet(feats_dir / "ligand_index.parquet")
    ik_to_row = dict(zip(lig_idx['inchi_key'], range(len(lig_idx))))
    
    pairs = pd.read_parquet(feats_dir / "scaffold_pairs_index.parquet")
    pool_U = pd.read_parquet(pu_dir / "pool_U_scaffold.parquet")
    
    train_pos = pairs[pairs['split'] == 'train']
    val_pos = pairs[pairs['split'] == 'val']
    
    # Downsample U pool to balance classes
    train_neg = pool_U.sample(min(len(train_pos), len(pool_U)), random_state=42)
    val_neg = pool_U[~pool_U.index.isin(train_neg.index)].sample(min(len(val_pos), len(pool_U) - len(train_neg)), random_state=42)
    
    train_df = pd.concat([train_pos, train_neg])
    y_train = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    
    val_df = pd.concat([val_pos, val_neg])
    y_val = np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))])
    
    # Extract ECFP4 features for train/val
    X_tr = ecfp[train_df['ligand_inchikey'].map(ik_to_row).fillna(0).astype(int).values]
    X_va = ecfp[val_df['ligand_inchikey'].map(ik_to_row).fillna(0).astype(int).values]
    
    # ── Train Real Models and Calibrate ───────────────────────────────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    lg.info("Fitting uncalibrated Random Forest baseline...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    y_prob_uncal = rf.predict_proba(X_va)[:, 1]
    
    lg.info("Fitting calibrated Random Forest wrapper...")
    cal_rf = CalibratedClassifierCV(
        estimator=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        method='sigmoid',
        cv=3
    )
    cal_rf.fit(X_tr, y_train)
    y_prob_cal = cal_rf.predict_proba(X_va)[:, 1]
    
    brier_u = brier_score_loss(y_val, y_prob_uncal)
    brier_c = brier_score_loss(y_val, y_prob_cal)
    
    lg.info(f"Uncalibrated Brier: {brier_u:.4f}")
    lg.info(f"Calibrated Brier:   {brier_c:.4f}")
    
    frac_u, mean_u = calibration_curve(y_val, y_prob_uncal, n_bins=10)
    frac_c, mean_c = calibration_curve(y_val, y_prob_cal, n_bins=10)
    
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(mean_u, frac_u, "s-", label=f"Uncalibrated RF (Brier = {brier_u:.3f})")
    plt.plot(mean_c, frac_c, "o-", label=f"Sigmoid Calibrated (Brier = {brier_c:.3f})")
    plt.ylabel("Fraction of Positives")
    plt.xlabel("Mean Predicted Probability")
    plt.title("Reliability Diagram")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "reliability_diagram.png", dpi=300)
    plt.savefig(fig_dir / "calibration_comparison.png", dpi=300)
    plt.close()
    
    lg.info("Phase 7 Calibration Analysis COMPLETE.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
