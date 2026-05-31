import pandas as pd
import numpy as np
import os
import sys
import importlib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --- Environment Guards (WinError 127 / OMP Conflict Fix) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONNOUSERSITE"] = "1"

if sys.platform == "win32":
    # 1. Strict Version Check
    if sys.version_info[:2] != (3, 11):
        print("\n" + "!" * 70)
        print(f"CRITICAL ERROR: WRONG PYTHON VERSION DETECTED ({sys.version.split()[0]})")
        print("This script MUST be run with Python 3.11 from the 'ifg26' environment.")
        print("!" * 70 + "\n")
        sys.exit(1)

    # 2. DLL Priority Fix (Force torch/lib to the front)
    env_base = r"C:\Users\Aadi Nair\miniconda3\miniconda4\envs\ifg26"
    torch_lib = os.path.join(env_base, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
# ------------------------------------------------------------

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

def compute_ecfp4(smiles_list):
    fps = []
    valid = []
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            fp = mfpgen.GetFingerprint(mol)
            fps.append(list(fp))
            valid.append(True)
        except:
            fps.append([0]*2048)
            valid.append(False)
    return np.array(fps), np.array(valid)

def run_cv(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    models = []
    
    has_xgb = importlib.util.find_spec("xgboost") is not None
    if not has_xgb:
        print("[WARNING] xgboost not found. Falling back to RandomForestClassifier.")
        
    for train_idx, test_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        if has_xgb:
            import xgboost as xgb
            clf = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, preds))
        models.append(clf)
    return np.mean(aucs), models

def main():
    print("Loading data for negative audit...")
    pmd_df = pd.read_parquet('data/negatives/pmd_relaxed.parquet')
    pool_p_df = pd.read_parquet('data/pu/pool_P_scaffold.parquet')
    ligands_df = pd.read_parquet('data/features/ligands_physchem.parquet')
    
    # Need to match P SMILES again to get the exact cohort
    smiles_map = {}
    for path in ['data/raw/mgtbind/mgtbind_compounds.csv', 'dataset/smiles_integrity_report.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            col = [c for c in df.columns if c.lower()=='smiles']
            if len(col) > 0:
                for sm in df[col[0]].dropna().unique():
                    try:
                        mol = Chem.MolFromSmiles(sm)
                        if mol: smiles_map[Chem.MolToInchiKey(mol)] = sm
                    except: pass
    if os.path.exists('data/raw/mgdb/mgdb_compounds.csv'):
        mgdb = pd.read_csv('data/raw/mgdb/mgdb_compounds.csv', skiprows=1)
        if 'Smiles' in mgdb.columns:
            for sm in mgdb['Smiles'].dropna().unique():
                try:
                    mol = Chem.MolFromSmiles(sm)
                    if mol: smiles_map[Chem.MolToInchiKey(mol)] = sm
                except: pass
                
    p_merged = pool_p_df.merge(ligands_df, left_on='ligand_inchikey', right_on='inchi_key', how='inner')
    p_merged['smiles'] = p_merged['ligand_inchikey'].map(smiles_map)
    p_merged = p_merged.dropna(subset=['smiles']).reset_index(drop=True)
    
    # Downsample P to match PMD size for balanced audit if needed, or just use all
    # The instructions say >= 1000 vs >= 1000. Let's just use all available valid P and PMD
    p_smiles = p_merged['smiles'].tolist()
    pmd_smiles = pmd_df['decoy_smiles'].tolist()
    
    labels = np.array([1]*len(p_smiles) + [0]*len(pmd_smiles))
    all_smiles = p_smiles + pmd_smiles
    
    # 1. Physchem Audit
    print("Computing Physchem descriptors...")
    props = []
    for s in all_smiles:
        try:
            mol = Chem.MolFromSmiles(s)
            from rdkit.Chem import Descriptors
            props.append({
                'MW': Descriptors.MolWt(mol),
                'cLogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotB': Descriptors.NumRotatableBonds(mol),
                'FormalCharge': Chem.GetFormalCharge(mol)
            })
        except:
            props.append({k: 0 for k in ['MW','cLogP','TPSA','HBD','HBA','RotB','FormalCharge']})
            
    df_physchem = pd.DataFrame(props)
    df_labels = pd.Series(labels)
    
    print("Running Physchem Audit 5-Fold CV...")
    physchem_auc, _ = run_cv(df_physchem, df_labels)
    print(f"Physchem AUROC: {physchem_auc:.4f}")
    
    # 2. ECFP4 Audit
    print("Computing ECFP4 fingerprints...")
    X_ecfp4, valid_mask = compute_ecfp4(all_smiles)
    df_ecfp4 = pd.DataFrame(X_ecfp4[valid_mask], columns=[f'bit_{i}' for i in range(2048)])
    labels_ecfp4 = df_labels[valid_mask].reset_index(drop=True)
    
    print("Running ECFP4 Audit 5-Fold CV...")
    ecfp4_auc, ecfp4_models = run_cv(df_ecfp4, labels_ecfp4)
    print(f"ECFP4 AUROC: {ecfp4_auc:.4f}")
    
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    
    reports = [
        f"# Phase 4E Negative Audit",
        f"\n**Physchem CV AUROC:** {physchem_auc:.4f} (Threshold: <= 0.70)",
        f"**ECFP4 CV AUROC:** {ecfp4_auc:.4f} (Threshold: <= 0.85)"
    ]
    
    shap_attempted = False
    shap_succeeded = False
    fallback_used = False
    warnings_list = []

    if ecfp4_auc > 0.85:
        has_shap = importlib.util.find_spec("shap") is not None
        has_xgb = importlib.util.find_spec("xgboost") is not None
        
        clf = ecfp4_models[0]
        
        if has_shap and has_xgb:
            print("ECFP4 AUROC > 0.85, extracting SHAP summary...")
            shap_attempted = True
            import shap
            try:
                explainer = shap.TreeExplainer(clf)
                # downsample for shap speed
                sample_idx = np.random.choice(len(df_ecfp4), min(1000, len(df_ecfp4)), replace=False)
                shap_values = explainer.shap_values(df_ecfp4.iloc[sample_idx])
                
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, df_ecfp4.iloc[sample_idx], show=False)
                plt.savefig('results/figures/phase4E_ecfp4_shap_summary.png', bbox_inches='tight')
                plt.close()
                reports.append("\n**Notice:** ECFP4 AUROC exceeded threshold. SHAP summary plot generated at `results/figures/phase4E_ecfp4_shap_summary.png`.")
                shap_succeeded = True
            except Exception as e:
                msg = f"SHAP TreeExplainer failed (likely XGBoost version incompatibility): {e}"
                print(msg)
                warnings_list.append("SHAP TreeExplainer failed")
                reports.append(f"\n**Notice:** ECFP4 AUROC exceeded threshold, but SHAP plots failed due to TreeExplainer incompatibility. Check docs.")
        else:
            msg = "ECFP4 AUROC > 0.85, but xgboost/shap not found. Skipping SHAP plot generation."
            print(msg)
            warnings_list.append("Missing xgboost or shap")
            reports.append("\n**Notice:** ECFP4 AUROC exceeded threshold, but SHAP plots were skipped due to missing dependencies (`xgboost`/`shap`).")
            
        if not shap_succeeded:
            print("Attempting to graph native feature importances as fallback...")
            try:
                importances = clf.feature_importances_
                indices = np.argsort(importances)[-20:]
                plt.figure(figsize=(10, 8))
                plt.title('Top 20 Feature Importances')
                plt.barh(range(len(indices)), importances[indices], align='center')
                plt.yticks(range(len(indices)), [f"bit_{i}" for i in indices])
                plt.xlabel('Relative Importance')
                plt.savefig('results/figures/phase4E_ecfp4_feature_importance.png', bbox_inches='tight')
                plt.close()
                fallback_used = True
                reports.append("\n**Fallback Used:** Plotted top 20 native feature importances in `results/figures/phase4E_ecfp4_feature_importance.png`.")
            except Exception as e:
                print(f"Fallback feature importance also failed: {e}")
                reports.append("\nExplainability step failed; core audit remains valid.")
                warnings_list.append("Native feature importances failed")
        
    with open('docs/phase4E_negative_audit.md', 'w') as f:
        f.write("\n".join(reports))
        
    print("Writing phase4E_negative_audit_status.json...")
    physchem_pass = bool(physchem_auc <= 0.70)
    ecfp4_pass = bool(ecfp4_auc <= 0.85)
    
    status = {
        "physchem_auroc": float(physchem_auc),
        "ecfp4_auroc": float(ecfp4_auc),
        "physchem_pass": physchem_pass,
        "ecfp4_pass": ecfp4_pass,
        "shap_attempted": shap_attempted,
        "shap_succeeded": shap_succeeded,
        "fallback_used": fallback_used,
        "warnings": warnings_list,
        "soft_fail": not (physchem_pass and ecfp4_pass),
        "reason": "negatives remain fingerprint-separable or physchem-separable" if not (physchem_pass and ecfp4_pass) else ""
    }
    
    os.makedirs('data/diagnostics', exist_ok=True)
    import json
    with open('data/diagnostics/phase4E_negative_audit_status.json', 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == '__main__':
    main()
