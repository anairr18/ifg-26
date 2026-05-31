"""
phase5B_negative_ladder.py
============================
IFG-26 Phase 5B — Negative Realism Ladder.

Evaluates model performance across 5 "tiers" of negative difficulty:
    1. Proxy Negatives (Random)
    2. PMD-v1 (Phase 2/3)
    3. Strict PMD (Phase 5A Internal)
    4. External PMD (Phase 5B External)
    5. PU Pool (Phase 4)

Models:
    - Logistic Regression (LR)
    - Random Forest (RF)
    - nnPU L0 (Loaded from .pt)
    - nnPU LP0 (Loaded from .pt)

Outputs:
    data/phase5B_negative_ladder_results.csv
    docs/phase5B_negative_ladder.md

Usage:
    python scripts/phase5B_negative_ladder.py [--resume]
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# --- Environment Guards (WinError 127 / OMP Conflict Fix) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONNOUSERSITE"] = "1"

if sys.platform == "win32":
    env_base = r"C:\Users\Aadi Nair\miniconda3\miniconda4\envs\ifg26"
    torch_lib = os.path.join(env_base, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
# ------------------------------------------------------------

import torch
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
_MGEN = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))

warnings.filterwarnings("ignore")

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5B_ladder")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5B_ladder.log", encoding="utf-8"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg

def get_fp(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = _MGEN.GetFingerprint(mol)
            arr = np.zeros((1,))
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
    except Exception: pass
    return None

def compute_recall_at_k(y_true, y_score, k=0.05):
    """Compute recall @ top k% of predictions."""
    n = len(y_true)
    k_idx = int(n * k)
    if k_idx == 0: k_idx = 1
    order = np.argsort(y_score)[::-1]
    top_y = y_true[order[:k_idx]]
    return np.sum(top_y) / np.sum(y_true) if np.sum(y_true) > 0 else 0

def compute_lift_at_k(y_true, y_score, k=0.05):
    """Compute lift @ top k% of predictions."""
    n = len(y_true)
    k_idx = int(n * k)
    if k_idx == 0: k_idx = 1
    order = np.argsort(y_score)[::-1]
    top_y = y_true[order[:k_idx]]
    precision_at_k = np.mean(top_y)
    prevalence = np.mean(y_true)
    return precision_at_k / prevalence if prevalence > 0 else 0

def eval_tier(X, y, tier_name, lg):
    """Evaluates LR and RF on a given tier via 5-fold CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    
    models = {
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    }
    
    for mname, model in models.items():
        auroc, auprc, recall5, lift5 = [], [], [], []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, 1]
            
            auroc.append(roc_auc_score(y_test, y_score))
            auprc.append(average_precision_score(y_test, y_score))
            recall5.append(compute_recall_at_k(y_test, y_score, k=0.05))
            lift5.append(compute_lift_at_k(y_test, y_score, k=0.05))
            
        lg.info(f"  [{tier_name}] {mname} | AUROC: {np.mean(auroc):.3f}")
        rows.append({
            "tier": tier_name, "model": mname,
            "auroc": np.mean(auroc), "auprc": np.mean(auprc),
            "recall@5": np.mean(recall5), "lift@5": np.mean(lift5)
        })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    out_path = ROOT / "data" / "phase5B_negative_ladder_results.csv"

    if args.resume and out_path.exists():
        lg.info("Output exists. Skipping.")
        return

    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 5B — Negative Realism Ladder  {ts}")
    lg.info("=" * 70)

    # 1. Load Positives
    curated_p2 = ROOT / "data" / "curated" / "phase2"
    pos_path = ROOT / "data/pu/pool_P_scaffold.parquet"
    if not pos_path.exists():
        lg.error(f"Positives not found at {pos_path}")
        return
    pos_df = pd.read_parquet(pos_path)
    
    # SMILES Join
    pos_smi_col = 'canonical_smiles' if 'canonical_smiles' in pos_df.columns else 'smiles'
    if pos_smi_col not in pos_df.columns:
        # Correct SMILES source is canonicalized_compounds.csv
        canon_path = ROOT / "dataset/phase1/canonicalized_compounds.csv"
        if canon_path.exists():
            df_canon = pd.read_csv(canon_path, low_memory=False, usecols=['inchi_key', 'canonical_smiles'])
            ik_map = dict(zip(df_canon['inchi_key'], df_canon['canonical_smiles']))
            ik_col_p = 'ligand_inchikey' if 'ligand_inchikey' in pos_df.columns else 'inchi_key'
            pos_df['smiles'] = pos_df[ik_col_p].map(ik_map)
            pos_smi_col = 'smiles'
        else:
            lg.error(f"Canonicalized compounds file not found at {canon_path}.")
            return
    
    pos_df = pos_df.dropna(subset=[pos_smi_col])
    lg.info(f"Loaded {len(pos_df)} positives.")
    pos_X = np.vstack([get_fp(s) for s in pos_df[pos_smi_col]])
    pos_y = np.ones(len(pos_X))
    pos_iks = set(pos_df['ligand_inchikey'].dropna())
    pos_smiles = set(pos_df[pos_smi_col].dropna())

    # 2. Define Core Tiers (External PMD v2 removed — documented separately as failure case)
    tier_candidates = {
        "Proxy": ROOT / "data/external_chembl_universe.parquet",
        "PMD-v1": ROOT / "dataset/phase2/test_scaffold.csv",
        "HARD-NEAR": ROOT / "data/phase5_hard_near_negatives.parquet",
        "PU Pool": ROOT / "data/pu/pool_U_scaffold.parquet"
    }
    TIER_ORDER = ["Proxy", "PMD-v1", "HARD-NEAR", "PU Pool"]

    all_results = []

    for name, path in tier_candidates.items():
        if not path.exists():
            lg.warning(f"Tier {name} file missing: {path}")
            continue
            
        lg.info(f"Evaluating Tier: {name}")
        if path.suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
            if "label" in df.columns: df = df[df["label"] == 0]
        else:
            df = pd.read_parquet(path)
            
        smi_col = next((c for c in ["canonical_smiles", "smiles"] if c in df.columns), None)
        if smi_col is None:
            # Try to join SMILES from canonicalized_compounds.csv
            ik_col = next((c for c in ["inchi_key", "ligand_inchikey"] if c in df.columns), None)
            canon_path = ROOT / "dataset/phase1/canonicalized_compounds.csv"
            if ik_col and canon_path.exists():
                lg.info(f"  Tier {name}: joining SMILES from canonicalized_compounds.csv...")
                df_canon = pd.read_csv(canon_path, low_memory=False, usecols=['inchi_key', 'canonical_smiles'])
                ik_map = dict(zip(df_canon['inchi_key'], df_canon['canonical_smiles']))
                df['smiles'] = df[ik_col].map(ik_map)
                smi_col = 'smiles'
            else:
                lg.warning(f"  Tier {name}: no SMILES column and no IK join possible. Skipping.")
                continue
        neg_df = df.dropna(subset=[smi_col])

        if name == "Proxy":
            ik_col = next((c for c in ["inchi_key", "ligand_inchikey"] if c in neg_df.columns), None)
            if ik_col: neg_df = neg_df[~neg_df[ik_col].isin(pos_iks)]
            neg_df = neg_df[~neg_df[smi_col].isin(pos_smiles)]
            neg_smiles = neg_df[smi_col].sample(n=min(len(pos_df), len(neg_df)), random_state=42).tolist()
        else:
            neg_smiles = neg_df[smi_col].head(len(pos_df)).tolist()
            
        lg.info(f"  Negatives: {len(neg_smiles)}")
        neg_X = np.vstack([get_fp(s) for s in neg_smiles])
        neg_y = np.zeros(len(neg_X))
        
        X = np.vstack([pos_X, neg_X])
        y = np.concatenate([pos_y, neg_y])
        
        all_results.extend(eval_tier(X, y, name, lg))

    # 4. Save & Plot
    res_df = pd.DataFrame(all_results)
    res_path = ROOT / "data/phase5_negative_ladder_core.csv"
    res_df.to_csv(res_path, index=False)
    lg.info(f"Core ladder results saved to {res_path}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        os.makedirs(ROOT / "figures", exist_ok=True)

        # Build summary: mean AUROC per tier
        summary = res_df.groupby('tier')['auroc'].mean().reset_index()
        summary['tier'] = pd.Categorical(summary['tier'], categories=TIER_ORDER, ordered=True)
        summary = summary.sort_values('tier')

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']  # Red -> Orange -> Yellow -> Green
        bars = ax.bar(summary['tier'], summary['auroc'], color=colors[:len(summary)], width=0.5, edgecolor='black', linewidth=0.8)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random baseline')
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel('AUROC (ECFP4, mean LR+RF)', fontsize=12)
        ax.set_title('IFG-26 Negative Realism Ladder (Core Tiers)', fontsize=13, fontweight='bold')
        for bar, (_, row) in zip(bars, summary.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{row["auroc"]:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(ROOT / "figures/negative_realism_ladder_core.png", dpi=150, bbox_inches='tight')
        plt.close()
        lg.info("Core ladder figure saved.")

        # Supplementary: include External PMD v2 for comparison
        ext_result = [
            {"tier": "External PMD v2", "model": "LR", "auroc": 0.970, "auprc": None},
            {"tier": "External PMD v2", "model": "RF", "auroc": 0.978, "auprc": None},
        ]
        all_tiers = ["Proxy", "External PMD v2", "PU Pool", "PMD-v1"]
        ext_df = pd.concat([res_df, pd.DataFrame(ext_result)], ignore_index=True)
        ext_summary = ext_df.groupby('tier')['auroc'].mean().reset_index()
        ext_summary['tier'] = pd.Categorical(ext_summary['tier'], categories=all_tiers, ordered=True)
        ext_summary = ext_summary.sort_values('tier')

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        bar_colors = ['#e74c3c', '#c0392b', '#2ecc71', '#f39c12']  # ext pmd2 is dark red = failure
        bars2 = ax2.bar(ext_summary['tier'], ext_summary['auroc'], color=bar_colors[:len(ext_summary)], width=0.5, edgecolor='black', linewidth=0.8)
        ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1)
        ax2.set_ylim(0.4, 1.05)
        ax2.set_ylabel('AUROC', fontsize=12)
        ax2.set_title('Negative Realism Ladder — External PMD v2 as Failure Case', fontsize=12, fontweight='bold')
        for bar, (_, row) in zip(bars2, ext_summary.iterrows()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{row["auroc"]:.3f}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(ROOT / "figures/negative_realism_ladder_external_failure.png", dpi=150, bbox_inches='tight')
        plt.close()
        lg.info("External failure figure saved.")

    except Exception as e:
        lg.warning(f"Plotting failed: {e}")

    with open(ROOT / "docs/phase5_core_ladder.md", "w", encoding="utf-8") as f:
        f.write("""# IFG-26 Phase 5 — Core Negative Realism Ladder

## Validated Benchmark Tiers

The core ladder uses three validated tiers representing a spectrum of negative realism:

| Tier | Description | AUROC (mean) |
|---|---|---|
""")
        for tier in TIER_ORDER:
            sub = res_df[res_df['tier'] == tier]
            if not sub.empty:
                f.write(f"| {tier} | {'Random drug-like from ChEMBL' if tier == 'Proxy' else 'Property-matched decoys (v1)' if tier == 'PMD-v1' else 'Unlabeled molecules from PU pool'} | {sub['auroc'].mean():.3f} |\n")
        f.write("""\n## Note on External PMD v2

External PMD v2 decoys (generated from ChEMBL with strict Tanimoto + property matching) achieved AUROC ≈ 0.97, identical to the Proxy tier. This indicates that external-universe decoys failed to produce realistic hard negatives and have been **excluded from the core benchmark ladder**. See `docs/phase5_external_pmd_failure_report.md` for full forensic analysis.
""")
    lg.info("Core ladder documentation saved.")
    lg.info("Phase 5B Core Ladder COMPLETE")

if __name__ == "__main__":
    main()

