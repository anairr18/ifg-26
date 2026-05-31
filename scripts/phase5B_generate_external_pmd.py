"""
phase5B_generate_external_pmd.py
==================================
IFG-26 Phase 5B — External Property-Matched Decoy Generator (OPTIMIZED).

High-performance implementation using:
    - Multi-processing for Fingerprinting
    - Vectorized NumPy for property filtering
    - RDKit BulkTanimotoSimilarity
    - Parallelized matching across 128-core TACC nodes

Outputs:
    data/phase5B_external_pmd_negatives.parquet
"""

import argparse
import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from multiprocessing import Pool

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

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from molecule_tracker import MoleculeTracker

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# PMD tolerances
TOL = {
    "mw_frac":   0.07,
    "logp_abs":  0.70,
    "tpsa_frac": 0.10,
    "hbd_int":   1,
    "hba_int":   1,
    "fsp3_abs":  0.10,
    "ring_int":  1,
    "tan_min":   0.20,
    "tan_max":   0.65,
}

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5B_pmd")
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5B_pmd.log", encoding="utf-8"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg

def get_fp_worker(smi):
    """Helper for parallel fingerprinting."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception: pass
    return None

def score_match_vectorized(pos_p, cand_props_batch, tan_batch):
    """Calculates combined distance score for a batch of candidates."""
    mw = max(pos_p["MolWt"], 1.0)
    tpsa = max(pos_p["TPSA"], 1.0)
    
    # Distance in property space (normalized by TOL)
    dists = []
    dists.append(np.abs(cand_props_batch[:, 0] - pos_p["MolWt"]) / (TOL["mw_frac"] * mw))
    dists.append(np.abs(cand_props_batch[:, 1] - pos_p["MolLogP"]) / TOL["logp_abs"])
    dists.append(np.abs(cand_props_batch[:, 2] - pos_p["TPSA"]) / (TOL["tpsa_frac"] * tpsa))
    dists.append(np.abs(cand_props_batch[:, 3] - pos_p["NumHDonors"]) / max(TOL["hbd_int"], 1))
    dists.append(np.abs(cand_props_batch[:, 4] - pos_p["NumHAcceptors"]) / max(TOL["hba_int"], 1))
    dists.append(np.abs(cand_props_batch[:, 5] - pos_p["FractionCSP3"]) / max(TOL["fsp3_abs"], 0.01))
    dists.append(np.abs(cand_props_batch[:, 6] - pos_p["RingCount"]) / max(TOL["ring_int"], 1))
    
    p_dist = np.mean(dists, axis=0)
    
    # Tanimoto penalty (prefer center of band)
    tan_center = (TOL["tan_min"] + TOL["tan_max"]) / 2
    tan_pen = np.abs(tan_batch - tan_center) / ((TOL["tan_max"] - TOL["tan_min"]) / 2)
    
    return 0.7 * p_dist + 0.3 * tan_pen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpus", type=int, default=32, help="Parallel cores (TACC nodes have 128)")
    args = parser.parse_args()

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    out_path = ROOT / "data" / "phase5B_external_pmd_negatives.parquet"

    if args.resume and out_path.exists():
        lg.info("Output exists. Skipping.")
        return

    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 5B — External PMD Generation (OPTIMIZED)  {ts}")
    lg.info("=" * 70)

    # 1. Load Universe
    uni_path = ROOT / "data" / "phase5_candidate_universe.parquet"
    if not uni_path.exists():
        lg.error("External universe not found. Run build_external_universe first.")
        sys.exit(1)
    uni_df = pd.read_parquet(uni_path)
    lg.info(f"External universe loaded: {len(uni_df)}")

    # 2. Load Positives
    curated_p2 = ROOT / "data" / "curated" / "phase2"
    pos_dfs = []
    for s in ["train_scaffold.csv", "val_scaffold.csv", "test_scaffold.csv"]:
        p = curated_p2 / s
        if p.exists(): pos_dfs.append(pd.read_csv(p, low_memory=False))
    if not pos_dfs:
        lg.error("No positives found.")
        sys.exit(1)
    pos_df = pd.concat(pos_dfs, ignore_index=True)
    
    pos_smiles_col = next((c for c in ["canonical_smiles", "canonical_isomeric_smiles", "smiles"] if c in pos_df.columns), None)
    ik_col = next((c for c in ["ligand_inchikey", "compound_inchi_key", "inchi_key"] if c in pos_df.columns), None)

    if pos_smiles_col is None:
        lg.error("No SMILES column found in split CSVs.")
        sys.exit(1)

    # Ensure InChIKeys and Valid Mols for Positives
    lg.info("Preprocessing positives...")
    pos_mols = []
    pos_iks = []
    for smi in pos_df[pos_smiles_col]:
        m = Chem.MolFromSmiles(str(smi))
        if m:
            pos_mols.append(m)
            pos_iks.append(Chem.MolToInchiKey(m))
    
    pos_data = pd.DataFrame({"mol": pos_mols, "inchi_key": pos_iks})
    pos_data = pos_data.drop_duplicates("inchi_key").reset_index(drop=True)
    lg.info(f"Unique positive ligands for matching: {len(pos_data)}")

    # 3. Parallel Pre-compute Candidate FPs
    lg.info(f"Parallel computing fingerprints for {len(uni_df)} candidates (cores={args.cpus})...")
    with Pool(args.cpus) as pool:
        cand_fps = pool.map(get_fp_worker, uni_df["smiles"].tolist())
    
    # Vectorize Candidate Properties
    prop_cols = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "FractionCSP3", "RingCount"]
    cand_props_mat = uni_df[prop_cols].values # (N_cand, 7)
    cand_iks = uni_df["inchi_key"].values

    # 4. Optimized Matching Loop
    pmd_records = []
    rejections = defaultdict(int)
    used_iks = set()
    lg.info(f"Starting O(N^2) matching: {len(pos_data)} vs {len(uni_df)} (total pairs: {len(pos_data)*len(uni_df):,})")

    for i, prow in pos_data.iterrows():
        if i > 0 and i % 500 == 0:
            lg.info(f"  Progress: {i}/{len(pos_data)} positives matched (PMDs so far: {len(pmd_records)})")
            
        pmol = prow["mol"]
        pik = prow["inchi_key"]
        
        # 1. Calc features for positive once
        pos_p = {
            "MolWt": Descriptors.MolWt(pmol),
            "MolLogP": Descriptors.MolLogP(pmol),
            "TPSA": rdMolDescriptors.CalcTPSA(pmol),
            "NumHDonors": rdMolDescriptors.CalcNumHBD(pmol),
            "NumHAcceptors": rdMolDescriptors.CalcNumHBA(pmol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(pmol),
            "RingCount": rdMolDescriptors.CalcNumRings(pmol),
        }
        pfp = AllChem.GetMorganFingerprintAsBitVect(pmol, 2, nBits=2048)
        
        # 2. Vectorized Property Filtering (Fast!)
        # Check all tolerances at once using NumPy
        mw_mask = np.abs(cand_props_mat[:, 0] - pos_p["MolWt"]) <= TOL["mw_frac"] * pos_p["MolWt"]
        logp_mask = np.abs(cand_props_mat[:, 1] - pos_p["MolLogP"]) <= TOL["logp_abs"]
        tpsa_limit = max(TOL["tpsa_frac"] * pos_p["TPSA"], 5.0)
        tpsa_mask = np.abs(cand_props_mat[:, 2] - pos_p["TPSA"]) <= tpsa_limit
        hbd_mask = np.abs(cand_props_mat[:, 3] - pos_p["NumHDonors"]) <= TOL["hbd_int"]
        hba_mask = np.abs(cand_props_mat[:, 4] - pos_p["NumHAcceptors"]) <= TOL["hba_int"]
        fsp3_mask = np.abs(cand_props_mat[:, 5] - pos_p["FractionCSP3"]) <= TOL["fsp3_abs"]
        ring_mask = np.abs(cand_props_mat[:, 6] - pos_p["RingCount"]) <= TOL["ring_int"]
        
        combined_mask = mw_mask & logp_mask & tpsa_mask & hbd_mask & hba_mask & fsp3_mask & ring_mask
        
        candidate_indices = np.where(combined_mask)[0]
        if len(candidate_indices) == 0:
            rejections["no_prop_match"] += 1
            continue
            
        # 3. Similarity Filtering for passing candidates
        valid_indices = []
        valid_tans = []
        
        # BulkTanimoto is 100x faster than looping
        batch_fps = [cand_fps[idx] for idx in candidate_indices]
        batch_tans = DataStructs.BulkTanimotoSimilarity(pfp, [f for f in batch_fps if f is not None])
        
        # Re-map valid Tans
        sim_pass_indices = []
        sim_pass_tans = []
        for j, tan in enumerate(batch_tans):
            if TOL["tan_min"] <= tan <= TOL["tan_max"]:
                orig_idx = candidate_indices[j]
                if cand_iks[orig_idx] not in used_iks:
                    sim_pass_indices.append(orig_idx)
                    sim_pass_tans.append(tan)
        
        if not sim_pass_indices:
            rejections["no_sim_match"] += 1
            continue
            
        # 4. Final Scoring (Vectorized score for survivors)
        sim_pass_indices = np.array(sim_pass_indices)
        sim_pass_tans = np.array(sim_pass_tans)
        
        scores = score_match_vectorized(pos_p, cand_props_mat[sim_pass_indices], sim_pass_tans)
        best_i = np.argmin(scores)
        best_idx = sim_pass_indices[best_i]
        
        used_iks.add(cand_iks[best_idx])
        crow = uni_df.iloc[best_idx]
        pmd_records.append({
            "inchi_key": cand_iks[best_idx],
            "smiles": crow["smiles"],
            "matched_positive_ik": pik,
            "tanimoto": sim_pass_tans[best_i],
            "distance_score": scores[best_i],
            **{c: crow[c] for c in prop_cols}
        })

    lg.info(f"Generation complete. PMDs matched: {len(pmd_records)}")

    # 5. Save Outputs
    pmd_df = pd.DataFrame(pmd_records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pmd_df.to_parquet(out_path, index=False)
    lg.info(f"Written: {out_path.name}")

    # Diagnostics
    diag_dir = ROOT / "data" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    with open(diag_dir / "phase5B_pmd_stats.json", "w") as f:
        json.dump({"pmd_count": len(pmd_records), "rejections": dict(rejections)}, f, indent=2)
    
if __name__ == "__main__":
    main()
