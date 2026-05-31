"""
phase7_shortcut_tests.py
========================
Phase 7 - Model Shortcut Tests
Trains models using restricted features (physchem only, ECFP4 only, Scaffold ID only, MW only).
Outputs:
    results/shortcut_test_results.csv
    figures/shortcut_comparison_barplot.png
"""

import os, sys, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

def setup_log():
    lg = logging.getLogger("phase7_shortcut")
    lg.setLevel(logging.INFO)
    lg.addHandler(logging.StreamHandler(sys.stdout))
    return lg

def main():
    lg = setup_log()
    lg.info("Phase 7 — Shortcut Tests (Real Out-of-Sample Evaluation)")
    
    res_dir = ROOT / "results"
    fig_dir = ROOT / "figures"
    feats_dir = ROOT / "data" / "features"
    pu_dir = ROOT / "data" / "pu"
    
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load Features and Indices ─────────────────────────────────────
    lg.info("Loading features and indices...")
    ecfp = np.load(feats_dir / "ligands_ecfp4.npy").astype(np.float32)
    lig_idx = pd.read_parquet(feats_dir / "ligand_index.parquet")
    ik_to_row = dict(zip(lig_idx['inchi_key'], range(len(lig_idx))))
    ik_to_smiles = dict(zip(lig_idx['inchi_key'], lig_idx['canonical_smiles']))
    
    phys_df = pd.read_parquet(feats_dir / "ligands_physchem.parquet")
    phys_cols = [c for c in phys_df.columns if c != "inchi_key"]
    phys_idx = phys_df.set_index("inchi_key")
    
    # ── Load Target ESM-2 Embeddings ──────────────────────────────────
    prot_df = pd.read_parquet(feats_dir / "protein_embeddings_esm2.parquet")
    emb_map = {r['uniprot_id']: r['embedding'].astype(np.float32) for _, r in prot_df.iterrows() if r['fetch_ok']}
    prot_dim = 1280
    
    def get_emb(uid):
        return emb_map.get(uid, np.zeros(prot_dim, dtype=np.float32))
        
    # ── Load Splits ───────────────────────────────────────────────────
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
    
    # Helper to clean/convert SMILES to ECFP
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    
    mfpgen = GetMorganGenerator(radius=2, fpSize=2048)
    
    def smiles_to_ecfp(smi: str) -> np.ndarray:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                return np.zeros(2048, dtype=np.float32)
            fp = mfpgen.GetFingerprint(mol)
            arr = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return np.zeros(2048, dtype=np.float32)
            
    def get_scaf_smiles(smi: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            return MurckoScaffoldSmiles(mol=mol) if mol else ""
        except Exception:
            return ""
            
    # ── Feature Extractors ────────────────────────────────────────────
    def extract_phys(df):
        rows = []
        for ik in df['ligand_inchikey']:
            if ik in phys_idx.index:
                rows.append(phys_idx.loc[ik, phys_cols].values)
            else:
                rows.append(np.zeros(len(phys_cols)))
        X = np.array(rows, dtype=np.float32)
        # Fill NaNs
        col_means = np.nanmean(X, axis=0) if len(X) > 0 else np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if j < len(col_means):
                X[mask, j] = col_means[j]
            else:
                X[mask, j] = 0.0
        return np.nan_to_num(X)
        
    def extract_ecfp(df):
        rows = df['ligand_inchikey'].map(ik_to_row).fillna(0).astype(int).values
        return ecfp[rows]
        
    def extract_mw(df):
        rows = []
        for ik in df['ligand_inchikey']:
            if ik in phys_idx.index:
                rows.append([phys_idx.loc[ik, 'MolWt']])
            else:
                rows.append([350.0])
        return np.nan_to_num(np.array(rows, dtype=np.float32))
        
    def extract_scaf(df):
        rows = []
        for ik in df['ligand_inchikey']:
            smi = ik_to_smiles.get(ik, "")
            scaf = get_scaf_smiles(smi)
            rows.append(smiles_to_ecfp(scaf))
        return np.array(rows, dtype=np.float32)
        
    def extract_full(df):
        l_f = extract_ecfp(df)
        e_f = np.stack(df['e3_uniprot_id'].apply(get_emb).values)
        t_f = np.stack(df['target_uniprot_id'].apply(get_emb).values)
        return np.hstack([l_f, e_f, t_f])
        
    # ── Train and Evaluate Baselines ──────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    feature_regimes = [
        ("Physchem only", extract_phys),
        ("ECFP4 only", extract_ecfp),
        ("Scaffold ID only", extract_scaf),
        ("Molecular weight only", extract_mw),
        ("Full Model", extract_full)
    ]
    
    results = []
    for name, extractor in feature_regimes:
        lg.info(f"Evaluating feature subset: {name}...")
        X_tr = extractor(train_df)
        X_va = extractor(val_df)
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        clf.fit(X_tr, y_train)
        
        prob = clf.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_val, prob) if len(np.unique(y_val)) > 1 else 0.5
        results.append({"Feature_Set": name, "AUROC": round(auc, 4)})
        lg.info(f"  {name} Out-of-Sample AUROC: {auc:.4f}")
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(res_dir / "shortcut_test_results.csv", index=False)
    
    # ── Generate Plot ─────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_res, x="Feature_Set", y="AUROC", palette="Reds_d")
    plt.axhline(0.5, color='black', linestyle='--', label='Random')
    plt.title("IFG-26 Feature Shortcut Vulnerability Test")
    plt.ylim(0.4, 1.0)
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "shortcut_comparison_barplot.png", dpi=300)
    plt.close()
    
    lg.info("Phase 7 Shortcut Tests COMPLETE.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
