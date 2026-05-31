"""
phase7_negative_quality_audit.py
================================
Phase 7 - Hard Negative Verification
Evaluates if decoys are distinguishable via LR/RF on physchem, ecfp4, combining features.
Verifies physchem AUROC <= 0.7, fingerprint <= 0.85
Outputs:
    results/negative_audit_summary.csv
"""

import os, sys, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

def setup_log():
    lg = logging.getLogger("phase7_neg")
    lg.setLevel(logging.INFO)
    lg.addHandler(logging.StreamHandler(sys.stdout))
    return lg

def main():
    lg = setup_log()
    lg.info("Phase 7 — Negative Quality Audit (Real Out-of-Sample CV Evaluation)")
    
    res_dir = ROOT / "results"
    feats_dir = ROOT / "data" / "features"
    neg_dir = ROOT / "data" / "negatives"
    
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load Positives and PMD Negatives ──────────────────────────────
    lg.info("Loading dataset files...")
    pmd_df = pd.read_parquet(neg_dir / "pmd_negatives.parquet")
    phys_df = pd.read_parquet(feats_dir / "ligands_physchem.parquet")
    ecfp_mat = np.load(feats_dir / "ligands_ecfp4.npy")
    lig_idx = pd.read_parquet(feats_dir / "ligand_index.parquet")
    pairs = pd.read_parquet(feats_dir / "scaffold_pairs_index.parquet")
    
    # Sample positives to match PMD decoy count
    pos_iks = pd.Series(pairs["ligand_inchikey"].unique())
    if len(pos_iks) > len(pmd_df):
        pos_iks = pos_iks.sample(len(pmd_df), random_state=42)
        
    ik_to_row = lig_idx.set_index("inchi_key")["row_idx"].to_dict()
    phys_idx = phys_df.set_index("inchi_key")
    
    # ── Physchem Features extraction ──────────────────────────────────
    phys_cols = [c for c in phys_df.columns if c != "inchi_key"]
    
    def get_phys(iks):
        rows = []
        for ik in iks:
            if ik in phys_idx.index:
                rows.append(phys_idx.loc[ik, phys_cols].values)
            else:
                rows.append(np.full(len(phys_cols), np.nan))
        return np.array(rows, dtype=np.float32)
        
    pos_phys = get_phys(pos_iks.tolist())
    pmd_phys = pmd_df[phys_cols].values.astype(np.float32) if all(c in pmd_df for c in phys_cols) \
               else np.column_stack([
                   pmd_df.get(c, pd.Series(np.nan, index=pmd_df.index)).values
                   for c in phys_cols]).astype(np.float32)
                   
    X_phys = np.vstack([pos_phys, pmd_phys])
    y = np.array([1]*len(pos_phys) + [0]*len(pmd_phys))
    
    # Clean NaNs
    col_means = np.nanmean(X_phys, axis=0)
    for j in range(X_phys.shape[1]):
        mask = np.isnan(X_phys[:, j])
        X_phys[mask, j] = col_means[j]
        
    # ── ECFP4 Features extraction ────────────────────────────────────
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    
    mfpgen = GetMorganGenerator(radius=2, fpSize=2048)
    
    def smiles_to_ecfp(smi: str) -> np.ndarray | None:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None: return None
            fp = mfpgen.GetFingerprint(mol)
            arr = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return None
            
    pos_ecfp_rows = []
    for ik in pos_iks:
        r = ik_to_row.get(ik, -1)
        if r >= 0:
            pos_ecfp_rows.append(ecfp_mat[r].astype(np.float32))
        else:
            pos_ecfp_rows.append(np.zeros(2048, dtype=np.float32))
            
    pmd_ecfp_rows = []
    for _, row in pmd_df.iterrows():
        arr = smiles_to_ecfp(str(row.get("canonical_smiles", "")))
        if arr is None:
            arr = np.zeros(2048, dtype=np.float32)
        pmd_ecfp_rows.append(arr)
        
    X_ecfp = np.vstack([pos_ecfp_rows, pmd_ecfp_rows])
    
    # Combined features
    X_combined = np.hstack([X_phys, X_ecfp])
    
    # ── 5-Fold Cross Validation Evaluation ─────────────────────────────
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    
    def evaluate_cv(X, y, model_type):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, va in kf.split(X, y):
            if model_type == "lr":
                clf = LogisticRegression(max_iter=1000, random_state=42)
            else:
                clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[va])[:, 1]
            scores.append(roc_auc_score(y[va], prob))
        return float(np.mean(scores))
        
    lg.info("Running CV evaluations on feature subsets...")
    
    phys_lr = evaluate_cv(X_phys, y, "lr")
    phys_rf = evaluate_cv(X_phys, y, "rf")
    ecfp_lr = evaluate_cv(X_ecfp, y, "lr")
    ecfp_rf = evaluate_cv(X_ecfp, y, "rf")
    comb_lr = evaluate_cv(X_combined, y, "lr")
    
    # Build results table matching original thresholds
    results = [
        {"Feature_Subset": "physchem", "Model": "Logistic Regression", "AUROC": round(phys_lr, 4), "Threshold": 0.7, "Pass": bool(phys_lr <= 0.70)},
        {"Feature_Subset": "physchem", "Model": "Random Forest", "AUROC": round(phys_rf, 4), "Threshold": 0.7, "Pass": bool(phys_rf <= 0.70)},
        {"Feature_Subset": "ecfp4", "Model": "Logistic Regression", "AUROC": round(ecfp_lr, 4), "Threshold": 0.85, "Pass": bool(ecfp_lr <= 0.85)},
        {"Feature_Subset": "ecfp4", "Model": "Random Forest", "AUROC": round(ecfp_rf, 4), "Threshold": 0.85, "Pass": bool(ecfp_rf <= 0.85)},
        {"Feature_Subset": "combined", "Model": "Logistic Regression", "AUROC": round(comb_lr, 4), "Threshold": 0.90, "Pass": bool(comb_lr <= 0.90)}
    ]
    df_res = pd.DataFrame(results)
    
    violation = False
    for _, row in df_res.iterrows():
        if not row["Pass"]:
            violation = True
            lg.warning(f"Violation Detected: {row['Feature_Subset']} using {row['Model']} scored {row['AUROC']} > {row['Threshold']}")
            
    if not violation:
        lg.info("All hard negative verification checks passed within expected AUROC limits.")
        
    df_res.to_csv(res_dir / "negative_audit_summary.csv", index=False)
    lg.info("Phase 7 Negative Quality Audit COMPLETE.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
