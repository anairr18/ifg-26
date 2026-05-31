"""
phase7_fingerprint_analysis.py
===============================
Phase 7 - Fingerprint Space Analysis
Calculates pairwise Tanimotos, NN similarity, and scaffold diversity.
Outputs:
    data/scaffold_overlap_matrix.csv
    results/fingerprint_statistics.csv
    figures/tanimoto_density_plot.png
    figures/scaffold_overlap_heatmap.png
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
    lg = logging.getLogger("phase7_fp")
    lg.setLevel(logging.INFO)
    lg.addHandler(logging.StreamHandler(sys.stdout))
    return lg

def main():
    lg = setup_log()
    lg.info("Phase 7 — Fingerprint Analysis (Real RDKit Evaluation)")
    
    res_dir = ROOT / "results" / "phase7"
    data_dir = ROOT / "data"
    fig_dir = ROOT / "figures"
    feats_dir = ROOT / "data" / "features"
    neg_dir = ROOT / "data" / "negatives"
    pu_dir = ROOT / "data" / "pu"
    
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load SMILES ───────────────────────────────────────────────────
    lg.info("Loading SMILES from datasets...")
    pairs = pd.read_parquet(feats_dir / "scaffold_pairs_index.parquet")
    lig_idx = pd.read_parquet(feats_dir / "ligand_index.parquet")
    ik_to_smiles = dict(zip(lig_idx['inchi_key'], lig_idx['canonical_smiles']))
    
    pos_smiles = [ik_to_smiles.get(ik, "") for ik in pairs["ligand_inchikey"].unique()]
    pos_smiles = [s for s in pos_smiles if s]
    
    pmd_df = pd.read_parquet(neg_dir / "pmd_negatives.parquet")
    pmd_smiles = pmd_df["canonical_smiles"].dropna().tolist()
    
    pool_U = pd.read_parquet(pu_dir / "pool_U_scaffold.parquet")
    pu_smiles = [ik_to_smiles.get(ik, "") for ik in pool_U["ligand_inchikey"].sample(min(300, len(pool_U)), random_state=42).tolist()]
    pu_smiles = [s for s in pu_smiles if s]
    
    # ── Convert to Morgan Fingerprints ────────────────────────────────
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    
    mfpgen = GetMorganGenerator(radius=2, fpSize=2048)
    
    def get_fps_and_scaffolds(smiles_list):
        fps = []
        scaffolds = []
        for s in smiles_list:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is None: continue
                fps.append(mfpgen.GetFingerprint(mol))
                scaf = MurckoScaffoldSmiles(mol=mol)
                if scaf:
                    scaffolds.append(scaf)
            except Exception:
                pass
        return fps, scaffolds
        
    lg.info("Computing fingerprints and Bemis-Murcko scaffolds...")
    pos_fps, pos_scaf = get_fps_and_scaffolds(pos_smiles)
    pmd_fps, pmd_scaf = get_fps_and_scaffolds(pmd_smiles)
    pu_fps, pu_scaf = get_fps_and_scaffolds(pu_smiles)
    
    # ── Compute Scaffold Entropy and Count ────────────────────────────
    def compute_entropy(scaffolds):
        if not scaffolds: return 0, 0.0
        counts = pd.Series(scaffolds).value_counts()
        p = counts / counts.sum()
        entropy = -np.sum(p * np.log2(p))
        return len(counts), float(entropy)
        
    pos_scaf_count, pos_entropy = compute_entropy(pos_scaf)
    pmd_scaf_count, pmd_entropy = compute_entropy(pmd_scaf)
    pu_scaf_count, pu_entropy = compute_entropy(pu_scaf)
    
    df_stats = pd.DataFrame({
        "Dataset": ["Positives", "PMD-v1", "PU Pool"],
        "scaffold_count": [pos_scaf_count, pmd_scaf_count, pu_scaf_count],
        "scaffold_entropy": [round(pos_entropy, 4), round(pmd_entropy, 4), round(pu_entropy, 4)]
    })
    df_stats.to_csv(res_dir / "fingerprint_statistics.csv", index=False)
    
    # ── Scaffold Overlap Matrix ───────────────────────────────────────
    # We construct a Tanimoto similarity cross-matrix between population fingerprints
    populations = {
        "Positives": pos_fps,
        "PMD-v1": pmd_fps,
        "PU Pool": pu_fps
    }
    pop_names = list(populations.keys())
    matrix = np.zeros((len(pop_names), len(pop_names)))
    
    for i, name_i in enumerate(pop_names):
        fps_i = populations[name_i]
        for j, name_j in enumerate(pop_names):
            fps_j = populations[name_j]
            sims = []
            for fp in fps_i[:100]:
                sims.extend(DataStructs.BulkTanimotoSimilarity(fp, fps_j[:100]))
            matrix[i, j] = np.mean(sims) if sims else 0.0
            
    overlaps = pd.DataFrame(matrix, columns=pop_names, index=pop_names)
    overlaps.to_csv(data_dir / "scaffold_overlap_matrix.csv")
    
    # Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(overlaps, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Scaffold Overlap Matrix (Mean Fingerprint Tanimoto)")
    plt.tight_layout()
    plt.savefig(fig_dir / "scaffold_overlap_heatmap.png", dpi=300)
    plt.close()
    
    # ── Tanimoto Density Plot ─────────────────────────────────────────
    pos_vs_pos = []
    n_pos = len(pos_fps)
    for i in range(min(300, n_pos)):
        for j in range(i + 1, min(300, n_pos)):
            pos_vs_pos.append(DataStructs.TanimotoSimilarity(pos_fps[i], pos_fps[j]))
            
    pos_vs_pmd = []
    for i in range(min(200, len(pos_fps))):
        sims = DataStructs.BulkTanimotoSimilarity(pos_fps[i], pmd_fps[:200])
        pos_vs_pmd.extend(sims)
        
    plt.figure(figsize=(8, 5))
    sns.kdeplot(pos_vs_pos, label="Positives vs Positives", fill=True, alpha=0.3)
    sns.kdeplot(pos_vs_pmd, label="Positives vs PMD Decoys", fill=True, alpha=0.3)
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Density")
    plt.title("Pairwise Tanimoto Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "tanimoto_density_plot.png", dpi=300)
    plt.close()
    
    lg.info("Phase 7 Fingerprint Analysis COMPLETE.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
