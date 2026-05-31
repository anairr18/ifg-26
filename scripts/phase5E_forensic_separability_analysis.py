"""
phase5E_forensic_separability_analysis.py
===========================================
IFG-26 Phase 5E — Forensic Separability Audit.

Tests if External PMD-v2 decoys are chemically distinguishable from positives.
Evaluation:
    - Models: Logistic Regression (LR), Random Forest (RF)
    - Features: Physchem descriptors (8), ECFP4 fingerprints (2048)
    - 5-fold Cross-Validation
    - Targets: Physchem AUROC <= 0.7, ECFP4 AUROC <= 0.85

Outputs:
    data/phase5E_forensic_separability.csv
    docs/phase5_external_pmd_v2_artifact_audit.md
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
_MGEN = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# --- Environment Guards ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONNOUSERSITE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore")

def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "FSP3": Descriptors.FractionCSP3(mol),
        "RotB": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol)
    }

def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    fp = _MGEN.GetFingerprint(mol)
    arr = np.zeros((1,), dtype=bool)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    print("IFG-26 Forensic Separability Audit v2")
    
    # 1. Load Data
    pos_path = ROOT / "data/pu/pool_P_scaffold.parquet"
    neg_path = ROOT / "data/phase5_external_pmd_v2.parquet"
    # Correct SMILES source
    canon_path = ROOT / "dataset/phase1/canonicalized_compounds.csv"
    
    if not pos_path.exists() or not neg_path.exists():
        print("Missing data files. Run previous phases first.")
        return
        
    df_p = pd.read_parquet(pos_path)
    df_n = pd.read_parquet(neg_path)
    
    # SMILES Join for positives
    if 'smiles' not in df_p.columns and 'canonical_smiles' not in df_p.columns:
        if canon_path.exists():
            print("Joining positives with canonicalized_compounds.csv for SMILES...")
            df_canon = pd.read_csv(canon_path, low_memory=False, usecols=['inchi_key', 'canonical_smiles'])
            ik_map = dict(zip(df_canon['inchi_key'], df_canon['canonical_smiles']))
            ik_col_p = 'ligand_inchikey' if 'ligand_inchikey' in df_p.columns else 'inchi_key'
            df_p['smiles'] = df_p[ik_col_p].map(ik_map)
        else:
            print(f"CRITICAL: Canonicalized compound file not found at {canon_path}.")
            return

    p_smi_col = 'canonical_smiles' if 'canonical_smiles' in df_p.columns else 'smiles'
    n_smi_col = 'canonical_smiles' if 'canonical_smiles' in df_n.columns else 'smiles'
    
    print(f"Loaded {len(df_p)} positives and {len(df_n)} PMD decoys.")
    
    # 2. Vectorize
    data = []
    fps = []
    labels = []
    
    print("Characterizing molecules...")
    # Positives
    df_p = df_p.dropna(subset=[p_smi_col])
    for smi in df_p[p_smi_col].unique():
        props = get_descriptors(smi)
        fp = get_fp(smi)
        if props and fp is not None:
            data.append(props)
            fps.append(fp)
            labels.append(1)
            
    # Negatives
    for smi in df_n[n_smi_col].unique():
        props = get_descriptors(smi)
        fp = get_fp(smi)
        if props and fp is not None:
            data.append(props)
            fps.append(fp)
            labels.append(0)
            
    X_prop = pd.DataFrame(data).values
    X_fp = np.array(fps)
    y = np.array(labels)
    
    # 3. Audit Loop
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for feature_set, X in [("Physchem", X_prop), ("ECFP4", X_fp)]:
        for mname, model in [
            ("LR", LogisticRegression(max_iter=1000)),
            ("RF", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
        ]:
            aucs = []
            for train_idx, val_idx in skf.split(X, y):
                model.fit(X[train_idx], y[train_idx])
                probs = model.predict_proba(X[val_idx])[:, 1]
                aucs.append(roc_auc_score(y[val_idx], probs))
            
            mean_auc = np.mean(aucs)
            print(f"  [{feature_set}] {mname} AUROC: {mean_auc:.4f}")
            results.append({"feature_set": feature_set, "model": mname, "auroc": mean_auc})

    # 4. Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(ROOT / "data/phase5E_forensic_separability.csv", index=False)
    
    # 5. Report
    doc_md = f"""# IFG-26 Phase 5 — External PMD-v2 Artifact Audit

## Summary of Separability
Evaluation of whether publication decoys are chemically distinct using simple classifiers.

| Feature Set | Model | AUROC | Status |
|---|---|---|---|
"""
    for _, r in res_df.iterrows():
        limit = 0.7 if r['feature_set'] == "Physchem" else 0.85
        status = "PASS" if r['auroc'] <= limit else "WARNING"
        doc_md += f"| {r['feature_set']} | {r['model']} | {r['auroc']:.3f} | {status} |\n"

    doc_md += f"\n## Interpretation\n- Target Physchem AUROC <= 0.70\n- Target ECFP4 AUROC <= 0.85"
    
    with open(ROOT / "docs/phase5_external_pmd_v2_artifact_audit.md", "w") as f:
        f.write(doc_md)

if __name__ == "__main__":
    main()
