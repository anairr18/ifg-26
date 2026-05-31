"""
phase7_distribution_audit.py
============================
Phase 7 - Distributional Dataset Audit
Compares physical properties between Positives, PMD-v1, PU Pool, and ChEMBL.
Outputs:
    results/phase7/distribution_statistics.csv
    figures/property_distribution_grid.png
    figures/wasserstein_heatmap.png
"""

import os, sys, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp, wasserstein_distance

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

def setup_log():
    lg = logging.getLogger("phase7_dist")
    lg.setLevel(logging.INFO)
    lg.addHandler(logging.StreamHandler(sys.stdout))
    return lg

def compute_props(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol: return None
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "Rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "Ring_count": rdMolDescriptors.CalcNumRings(mol),
            "FSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "Aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
            "Heteroatom_fraction": rdMolDescriptors.CalcNumHeteroatoms(mol) / max(1, mol.GetNumAtoms())
        }
    except: return None

def main():
    lg = setup_log()
    lg.info("Phase 7 — Distribution Audit (Real Dataset Property Evaluation)")
    
    res_dir = ROOT / "results" / "phase7"
    fig_dir = ROOT / "figures"
    feats_dir = ROOT / "data" / "features"
    neg_dir = ROOT / "data" / "negatives"
    pu_dir = ROOT / "data" / "pu"
    
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load Physchem Descriptors and Splits ──────────────────────────
    lg.info("Loading pre-computed physicochemical descriptors...")
    phys_df = pd.read_parquet(feats_dir / "ligands_physchem.parquet")
    phys_cols = [c for c in phys_df.columns if c != "inchi_key"]
    phys_idx = phys_df.set_index("inchi_key")
    
    pairs = pd.read_parquet(feats_dir / "scaffold_pairs_index.parquet")
    pmd_df = pd.read_parquet(neg_dir / "pmd_negatives.parquet")
    pool_U = pd.read_parquet(pu_dir / "pool_U_scaffold.parquet")
    
    pos_iks = pairs["ligand_inchikey"].unique()
    pmd_iks = pmd_df["inchi_key"].dropna().unique() if "inchi_key" in pmd_df.columns \
              else pmd_df["canonical_smiles"].dropna().unique() # fallback
              
    pu_iks = pool_U["ligand_inchikey"].dropna().sample(min(1000, len(pool_U)), random_state=42).unique()
    
    # Helper to fetch property vectors
    def get_population_props(iks):
        rows = []
        for ik in iks:
            if ik in phys_idx.index:
                rows.append(phys_idx.loc[ik, phys_cols].values)
        X = np.array(rows, dtype=np.float32)
        if len(X) > 0:
            col_means = np.nanmean(X, axis=0)
            for j in range(X.shape[1]):
                mask = np.isnan(X[:, j])
                X[mask, j] = col_means[j]
        return pd.DataFrame(X, columns=phys_cols)
        
    lg.info("Extracting properties for each population...")
    pos_props = get_population_props(pos_iks)
    pmd_props = pmd_df[phys_cols].dropna().astype(np.float32) if all(c in pmd_df for c in phys_cols) \
                else get_population_props(pmd_iks)
    pu_props = get_population_props(pu_iks)
    
    # Map subset of features we wish to audit
    props = ["MolWt", "LogP", "TPSA", "NumHBD", "NumHBA", "NumRotatableBonds", "NumRings", "FractionCSP3", "NumAromaticRings"]
    props = [p for p in props if p in pos_props.columns]
    
    # ── Compute K-S Statistics and Wasserstein Distances ─────────────
    lg.info("Computing real distributional distance statistics...")
    results = []
    for prop in props:
        v1 = pos_props[prop].values
        v2 = pmd_props[prop].values
        if len(v1) > 0 and len(v2) > 0:
            ks_stat, p_val = ks_2samp(v1, v2)
            wd = wasserstein_distance(v1, v2)
            results.append({
                "Property": prop,
                "Dataset1": "Positives",
                "Dataset2": "PMD-v1",
                "KS_Stat": round(ks_stat, 4),
                "KS_pval": round(p_val, 6),
                "Wasserstein": round(wd, 4)
            })
            
    df_stats = pd.DataFrame(results)
    df_stats.to_csv(res_dir / "distribution_statistics.csv", index=False)
    
    # ── Generate Property Distribution Grid ───────────────────────────
    plt.figure(figsize=(15, 10))
    for i, prop in enumerate(props, 1):
        plt.subplot(3, 3, i)
        sns.kdeplot(pos_props[prop].values, label="Positives", fill=True, alpha=0.3)
        sns.kdeplot(pmd_props[prop].values, label="PMD Decoys", fill=True, alpha=0.3)
        plt.title(prop)
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "property_distribution_grid.png", dpi=300)
    plt.close()
    
    # ── Wasserstein Matrix (Molecular Weight / MolWt) ─────────────────
    pop_names = ["Positives", "PMD-v1", "PU Pool"]
    pop_data = [pos_props["MolWt"].values, pmd_props["MolWt"].values, pu_props["MolWt"].values]
    
    mat = np.zeros((len(pop_names), len(pop_names)))
    for i in range(len(pop_names)):
        for j in range(len(pop_names)):
            mat[i, j] = wasserstein_distance(pop_data[i], pop_data[j])
            
    overlaps = pd.DataFrame(mat, columns=pop_names, index=pop_names)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(overlaps, annot=True, cmap="Oranges", fmt=".3f")
    plt.title("Wasserstein Distance Matrix (MolWt)")
    plt.tight_layout()
    plt.savefig(fig_dir / "wasserstein_heatmap.png", dpi=300)
    plt.close()
    
    lg.info("Phase 7 Distribution Audit COMPLETE.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
