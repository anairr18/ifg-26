import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
import sys

# Paths
ROOT = Path("c:/Users/Aadi Nair/Downloads/IFG26")
FEATS_DIR = ROOT / "data/features"
PU_DIR = ROOT / "data/pu"
BUNDLE_DIR = ROOT / "data/phase8_eval_bundles"
MGDB_PATH = ROOT / "dataset/phase1/mgdb_compounds_canonicalized.csv"

def compute_ecfp(smiles, radius=2, nbits=2048):
    if not isinstance(smiles, str) or not smiles or smiles.lower() == "nan":
        return np.zeros(nbits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros(nbits, dtype=np.float32)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fp = mfpgen.GetFingerprint(mol)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_real_metrics():
    print("Loading positive training data...")
    ecfp_cache = np.load(FEATS_DIR / "ligands_ecfp4.npy")
    pairs = pd.read_parquet(FEATS_DIR / "scaffold_pairs_index.parquet", engine='pyarrow')
    train_pos = pairs[pairs['split'] == 'train']
    X_pos = ecfp_cache[train_pos['ligand_feature_row'].values].astype(np.float32)
    
    print("Generating proxy negatives from MGDB...")
    mgdb = pd.read_csv(MGDB_PATH, low_memory=False)
    pos_iks = set(pairs['ligand_inchikey'].unique())
    neg_pool = mgdb[~mgdb['inchi_key'].isin(pos_iks)].dropna(subset=['canonical_smiles'])
    mgdb_negs = neg_pool.sample(len(X_pos), replace=True, random_state=42)
    
    neg_fps = [compute_ecfp(s) for s in mgdb_negs['canonical_smiles']]
    X_neg = np.array(neg_fps, dtype=np.float32)
    
    X_train = np.vstack([X_pos, X_neg])
    y_train = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)
    
    regimes = [
        ("S1_Proxy", "Proxy"),
        ("S2_PMDv1", "PMD-v1"),
        ("S3_Scaffold", "Scaffold Holdout"),
        ("S4_E3Holdout", "Protein Holdout (E3)"),
        ("S5_TargetHoldout", "Protein Holdout (Tgt)"),
        ("S6_PURanking", "PU Ranking")
    ]
    
    results = []
    for r_code, r_name in regimes:
        p = BUNDLE_DIR / f"{r_code}.parquet"
        if not p.exists(): continue
        
        print(f"Evaluating on {r_name}...")
        df_eval = pd.read_parquet(p, engine='pyarrow')
        
        if 'ligand_feature_row' in df_eval.columns and (df_eval['ligand_feature_row'] >= 0).any():
            rows = df_eval['ligand_feature_row'].values
            X_eval = []
            for i, row_idx in enumerate(rows):
                if pd.notna(row_idx) and int(row_idx) >= 0 and int(row_idx) < len(ecfp_cache):
                    X_eval.append(ecfp_cache[int(row_idx)])
                else:
                    smi = ""
                    for c in ['smiles', 'canonical_smiles', 'canonical_isomeric_smiles']:
                        if c in df_eval.columns: smi = df_eval.iloc[i][c]; break
                    X_eval.append(compute_ecfp(smi))
            X_eval = np.array(X_eval, dtype=np.float32)
        else:
            smi_col = None
            for c in ['smiles', 'canonical_smiles', 'canonical_isomeric_smiles']:
                if c in df_eval.columns: smi_col = c; break
            if smi_col:
                X_eval = np.array([compute_ecfp(s) for s in df_eval[smi_col]], dtype=np.float32)
            else: continue
                
        y_eval = df_eval['label'].values
        preds = model.predict_proba(X_eval)[:, 1]
        auc = roc_auc_score(y_eval, preds)
        
        eval_res = pd.DataFrame({'y': y_eval, 's': preds}).sort_values('s', ascending=False)
        top_5 = max(1, int(len(eval_res) * 0.05))
        rec5 = eval_res.iloc[:top_5]['y'].sum() / eval_res['y'].sum() if eval_res['y'].sum() > 0 else 0
        
        results.append({"Regime": r_name, "AUROC": auc, "Recall@5%": rec5})
        print(f"{r_name}: AUROC={auc:.4f}, Recall@5%={rec5:.4f}")

    print("\n--- FINAL QUANTITATIVE SUMMARY (REAL DATA) ---")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_markdown(index=False))
    summary_df.to_csv(ROOT / "results" / "real_stress_test_metrics_final.csv", index=False)

if __name__ == "__main__":
    get_real_metrics()
