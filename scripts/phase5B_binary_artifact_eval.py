"""
phase5B_binary_artifact_eval.py
=================================
IFG-26 Phase 5B — Binary Artifact Audit.

Evaluates whether positive ligands and external PMD negatives remain
separable by simple physical properties or fingerprints. High AUROC
indicates the presence of "leakage" or artifacts in negative construction.

Metrics:
    - Physchem AUROC (MW, logP, TPSA, etc.)
    - ECFP4 AUROC

Models:
    - Logistic Regression
    - Random Forest
    - 5-fold Cross-Validation

Outputs:
    data/phase5B_binary_eval.csv
    docs/phase5B_negative_audit.md

Usage:
    python scripts/phase5B_binary_artifact_eval.py [--resume]
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))

warnings.filterwarnings("ignore")

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5B_audit")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5B_audit.log", encoding="utf-8"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg

def get_fp(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((1,))
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    out_path = ROOT / "data" / "phase5B_binary_eval.csv"

    if args.resume and out_path.exists():
        lg.info("Output exists. Skipping.")
        return

    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 5B — Binary Artifact Audit  {ts}")
    lg.info("=" * 70)

    # 1. Load Positives
    curated_p2 = ROOT / "data" / "curated" / "phase2"
    pos_dfs = []
    for s in ["train_scaffold.csv", "val_scaffold.csv", "test_scaffold.csv"]:
        p = curated_p2 / s
        if p.exists(): pos_dfs.append(pd.read_csv(p, low_memory=False))
    if not pos_dfs:
        lg.error("No positives found.")
        sys.exit(1)
    pos_df = pd.concat(pos_dfs, ignore_index=True)
    pos_smi_col = next((c for c in ["canonical_smiles", "smiles"] if c in pos_df.columns), None)
    pos_df = pos_df.dropna(subset=[pos_smi_col]).drop_duplicates(pos_smi_col)
    pos_df["label"] = 1

    # 2. Load External PMDs
    pmd_path = ROOT / "data" / "phase5B_external_pmd_negatives.parquet"
    if not pmd_path.exists():
        lg.error("External PMDs not found. Run generate_external_pmd first.")
        sys.exit(1)
    pmd_df = pd.read_parquet(pmd_path)
    pmd_df["label"] = 0

    lg.info(f"Positives: {len(pos_df)} | External PMDs: {len(pmd_df)}")

    # 3. Features
    prop_cols = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "FractionCSP3", "RingCount"]
    
    # We need to make sure positives have these props. 
    # Usually they would be curated, but let's recompute if missing for safety or use tracker.
    # For now, assume they might need computation since splits might not have all 7 cols.
    from rdkit.Chem import Descriptors, rdMolDescriptors
    def compute_all(smi):
        mol = Chem.MolFromSmiles(smi)
        if not mol: return None
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            rdMolDescriptors.CalcNumRings(mol)
        ]

    lg.info("Computing features for positives...")
    pos_props = []
    pos_fps = []
    for s in pos_df[pos_smi_col]:
        p = compute_all(s)
        f = get_fp(s)
        pos_props.append(p)
        pos_fps.append(f)
    
    pos_df["props"] = pos_props
    pos_df["fp"] = pos_fps
    pos_df = pos_df.dropna(subset=["props", "fp"])

    lg.info("Extracting features for PMDs...")
    pmd_props = pmd_df[prop_cols].values.tolist()
    pmd_fps = [get_fp(s) for s in pmd_df["smiles"]]
    pmd_df["props"] = pmd_props
    pmd_df["fp"] = pmd_fps
    pmd_df = pmd_df.dropna(subset=["props", "fp"])

    # Combine
    X_prop = np.vstack([np.array(pos_df["props"].tolist()), np.array(pmd_df["props"].tolist())])
    X_fp = np.vstack([np.array(pos_df["fp"].tolist()), np.array(pmd_df["fp"].tolist())])
    y = np.concatenate([pos_df["label"].values, pmd_df["label"].values])

    # 4. Evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    def run_eval(X, y, name, model_name):
        if model_name == "LR":
            X_scaled = StandardScaler().fit_transform(X)
            clf = LogisticRegression(max_iter=1000)
            scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="roc_auc")
        else:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
        
        lg.info(f"  {name} | {model_name} | AUROC: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        return {
            "feature_set": name,
            "model": model_name,
            "auroc_mean": np.mean(scores),
            "auroc_std": np.std(scores)
        }

    results.append(run_eval(X_prop, y, "Physchem", "LR"))
    results.append(run_eval(X_prop, y, "Physchem", "RF"))
    results.append(run_eval(X_fp, y, "ECFP4", "LR"))
    results.append(run_eval(X_fp, y, "ECFP4", "RF"))

    # 5. Scientific Status
    max_auroc = max(r["auroc_mean"] for r in results)
    if max_auroc > 0.85:
        status = "fail"
        reason = "Extremely separable; negatives likely carry construction artifacts."
    elif max_auroc > 0.70:
        status = "warning"
        reason = "Moderately separable; performance may be over-inflated."
    else:
        status = "pass"
        reason = "Well-matched; models must learn non-trivial features."

    lg.info(f"Scientific Status: {status.upper()} - {reason}")

    # Save
    res_df = pd.DataFrame(results)
    res_df["scientific_status"] = status
    res_df["status_reason"] = reason
    res_df.to_csv(out_path, index=False)
    lg.info(f"Written: {out_path.relative_to(ROOT)}")

    # Documentation
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    rows = "\n".join([f"| {r['feature_set']} | {r['model']} | {r['auroc_mean']:.3f} |" for r in results])
    
    status_color = "red" if status == "fail" else "orange" if status == "warning" else "green"
    
    doc_md = f"""# IFG-26 Phase 5B — Negative Audit

_Generated: {ts}_

## Scientific Status: <span style="color:{status_color}">{status.upper()}</span>

**Reason**: {reason}

## Separability AUROC (5-fold CV)

| Feature Set | Model | Mean AUROC |
|---|---|---|
{rows}

## Interpretation
A high AUROC (especially >0.85) indicates that the "difficult" PMD negatives are still
easily distinguishable from positives using basic descriptors or structural fingerprints.
In such cases, reported model performance on molecular glue prediction may be
reflecting construction artifacts rather than biological binding signal.
"""
    with open(docs_dir / "phase5B_negative_audit.md", "w", encoding="utf-8") as f:
        f.write(doc_md)
    lg.info("Written: docs/phase5B_negative_audit.md")

    lg.info("=" * 70)
    lg.info(f"Phase 5B Audit COMPLETE: {status.upper()}")
    lg.info("=" * 70)

if __name__ == "__main__":
    main()
