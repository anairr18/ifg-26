import os
import glob
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def get_fps(smiles_list):
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s))
        if m:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    return fps

def compute_nn_similarities(query_fps, ref_fps):
    """Calculates max similarity to any reference for each query."""
    max_sims = []
    for q in query_fps:
        sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
        max_sims.append(max(sims))
    return max_sims

def main():
    root = Path(__file__).resolve().parent.parent
    lg_dir = root / "logs"
    lg_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Training Positives (Gold standard for distance)
    pos_dir = root / "data" / "curated" / "phase1"
    pos_paths = [pos_dir / "mgdb_compounds_canonicalized.csv", pos_dir / "mgtbind_compounds_canonicalized.csv"]
    pos_smis = []
    for p in pos_paths:
        if p.exists():
            pos_smis.extend(pd.read_csv(p)["canonical_smiles"].tolist())
    
    # 2. Load External Universe
    uni_path = root / "data" / "phase5B_external_candidate_universe.parquet"
    if not uni_path.exists():
        print("External universe not found.")
        return
    uni_df = pd.read_parquet(uni_path)
    
    print(f"Computing OOD distance for {len(uni_df)} candidates vs {len(pos_smis)} training positives...")
    
    # Sample if too large for speed (10k is plenty for distribution)
    u_smi_sample = uni_df["smiles"].sample(min(10000, len(uni_df)), random_state=42).tolist()
    
    pos_fps = get_fps(pos_smis)
    u_fps = get_fps(u_smi_sample)
    
    nn_sims = compute_nn_similarities(u_fps, pos_fps)
    
    # 3. Save & Plot
    diag_dir = root / "data" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    res_df = pd.DataFrame({"nn_tanimoto": nn_sims})
    res_df.to_csv(diag_dir / "ood_chemical_distance.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    plt.hist(nn_sims, bins=50, alpha=0.75, color='royalblue', edgecolor='black')
    plt.axvline(np.mean(nn_sims), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(nn_sims):.3f}')
    plt.title("Chemical OOD Distribution (External Universe vs Training Set)")
    plt.xlabel("Nearest-Neighbor Tanimoto Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "ood_similarity_distribution.png", dpi=300)
    print(f"OOD Figure saved: figures/ood_similarity_distribution.png")

    # Final stats
    stats = {
        "mean_similarity": float(np.mean(nn_sims)),
        "median_similarity": float(np.median(nn_sims)),
        "pct_above_0.4": float(np.mean(np.array(nn_sims) > 0.4) * 100),
        "pct_above_0.6": float(np.mean(np.array(nn_sims) > 0.6) * 100)
    }
    with open(diag_dir / "ood_metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
