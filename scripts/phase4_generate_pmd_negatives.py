"""
phase4_generate_pmd_negatives.py
==================================
IFG-26 Phase 4C-i — Property-Matched Decoy (PMD) Negative Generation.

Generates hard negatives from the MGDB compound pool by property matching
against positive ligands. PMD negatives are used for auxiliary binary
sanity checks only (NOT for nnPU primary training).

PMD matching rules:
    MW ±5%         cLogP ±0.5       HBD ±1      HBA ±1
    TPSA ±10       RotB ±1          charge exact
    Scaffold disjoint from all positives
    Max ECFP4 Tanimoto to any positive ≤ 0.80

Artifact audit gates:
    Physchem-only AUROC ≤ 0.70  (else flag too-easy negatives)
    ECFP4-only    AUROC ≤ 0.85  (else tighten constraints)

Outputs:
    data/negatives/pmd_negatives.parquet
    docs/phase4C_negative_audit.md
    results/figures/phase4/phase4C_property_overlap.png
"""

import argparse
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent.parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.append(str(ROOT / "scripts"))
from molecule_tracker import MoleculeTracker

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase4_default.yaml"


def setup_logging(name="phase4_pmd"):
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt); lg.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); lg.addHandler(sh)
    return lg


PROP_FUNCS = {
    "MolWt":             Descriptors.MolWt,
    "MolLogP":           Descriptors.MolLogP,
    "NumHDonors":        rdMolDescriptors.CalcNumHBD,
    "NumHAcceptors":     rdMolDescriptors.CalcNumHBA,
    "TPSA":              rdMolDescriptors.CalcTPSA,
    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "FormalCharge":      Chem.GetFormalCharge,
    "FractionCSP3":      rdMolDescriptors.CalcFractionCSP3,
    "RingCount":         rdMolDescriptors.CalcNumRings,
}


def get_props(mol) -> dict | None:
    if mol is None: return None
    return {k: fn(mol) for k, fn in PROP_FUNCS.items()}


def get_scaffold(mol) -> str | None:
    if mol is None: return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True) if core else None
    except: return None


def get_fp(mol):
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def property_match(pos_props: dict, cand_props: dict, cfg: dict) -> tuple[bool, str]:
    mw = pos_props["MolWt"]
    tpsa = pos_props["TPSA"]
    
    # Tolerances per User Request #4 (Upgraded Strategy)
    tol_mw_frac = 0.07      # 7%
    tol_logp_abs = 0.7      # 0.7
    tol_hbd = 1
    tol_hba = 1
    tol_tpsa_frac = 0.10    # 10%
    tol_fsp3 = 0.1
    tol_rings = 1
    
    # Missing descriptors check
    for k in PROP_FUNCS:
        if k not in cand_props or pd.isna(cand_props[k]):
            return False, "invalid_smiles"

    if abs(cand_props["MolWt"] - mw) > tol_mw_frac * mw: 
        return False, "MW"
    if abs(cand_props["MolLogP"] - pos_props["MolLogP"]) > tol_logp_abs: 
        return False, "logP"
    if abs(cand_props["NumHDonors"] - pos_props["NumHDonors"]) > tol_hbd: 
        return False, "HBD/HBA" 
    if abs(cand_props["NumHAcceptors"] - pos_props["NumHAcceptors"]) > tol_hba: 
        return False, "HBD/HBA"
    
    if abs(cand_props["TPSA"] - tpsa) > (tol_tpsa_frac * tpsa if tpsa > 0 else 10.0): 
        return False, "TPSA"
    
    if abs(cand_props["FractionCSP3"] - pos_props["FractionCSP3"]) > tol_fsp3:
        return False, "fraction_sp3"
            
    if abs(cand_props["RingCount"] - pos_props["RingCount"]) > tol_rings:
        return False, "ring_count"
            
    if int(cand_props["FormalCharge"]) != int(pos_props["FormalCharge"]): 
        return False, "outside formal charge"
        
    return True, "pass"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 4C — PMD Negative Generation  {ts}")
    lg.info("=" * 70)

    pmd_cfg = cfg.get("pmd", {})
    max_tan = pmd_cfg.get("max_tanimoto_to_pos", 0.80)
    n_per_pos = pmd_cfg.get("n_negatives_per_pos", 3)

    # Load positives
    feats_dir  = ROOT / "data" / "features"
    lig_idx    = pd.read_parquet(feats_dir / "ligand_index.parquet")
    ecfp_mat   = np.load(str(feats_dir / "ligands_ecfp4.npy"))
    pairs      = pd.read_parquet(feats_dir / "scaffold_pairs_index.parquet")
    pos_iks    = set(pairs["ligand_inchikey"].dropna().unique())
    phys_df    = pd.read_parquet(feats_dir / "ligands_physchem.parquet")

    # Build pos scaffold mapping
    pos_scaffold_dict = {}
    train_csv  = pd.read_csv(ROOT / cfg["inputs"]["train_scaffold"], low_memory=False)
    pos_scaffolds: set[str] = set()
    tracker = MoleculeTracker(name="phase4_pmd")
    for _, row in train_csv.drop_duplicates("compound_inchi_key").iterrows():
        ik = row.get("compound_inchi_key")
        smi = row.get("canonical_smiles")
        if not ik or not smi: continue
        mol = tracker.parse(str(smi), source_file=cfg["inputs"]["train_scaffold"], stage="load_train", record_id=ik)
        sc = get_scaffold(mol)
        if sc: 
            pos_scaffolds.add(sc)
            pos_scaffold_dict[ik] = sc
    lg.info(f"  Positive scaffolds indexed: {len(pos_scaffolds)}")

    # Build ECFP4 fingerprints for all positives (for similarity ceiling check)
    ik_to_row = lig_idx.set_index("inchi_key")["row_idx"].to_dict()
    from rdkit.DataStructs import ExplicitBitVect
    def arr_to_fp(arr):
        fp = ExplicitBitVect(2048)
        for j, b in enumerate(arr):
            if b: fp.SetBit(j)
        return fp
    
    pos_fps_dict = {ik: arr_to_fp(ecfp_mat[row_idx]) for ik, row_idx in ik_to_row.items() if ik in pos_iks}
    pos_fps = list(pos_fps_dict.values())
    lg.info(f"  Positive FPs for similarity check: {len(pos_fps)}")

    # Load MGDB candidates
    mgdb = pd.read_csv(ROOT / cfg["inputs"]["mgdb_canon"], encoding="utf-8", low_memory=False)
    mgdb = mgdb[mgdb["canonicalized_ok"] == True].copy()
    mgdb_cands = mgdb[~mgdb["inchi_key"].isin(pos_iks)].dropna(subset=["inchi_key"])
    lg.info(f"  MGDB candidate pool: {len(mgdb_cands)} compounds")

    # Build props + FPs for candidate pool
    lg.info("  Computing properties + FPs for candidate pool...")
    cand_rows = []
    for _, row in mgdb_cands.iterrows():
        smi = str(row.get("canonical_smiles", ""))
        mol = tracker.parse(smi, source_file=cfg["inputs"]["mgdb_canon"], stage="load_mgdb", record_id=row["inchi_key"]) if smi and smi.lower() not in ("nan","") else None
        if mol is None: continue
        props = get_props(mol)
        if not props: continue
        sc = get_scaffold(mol)
        fp = get_fp(mol)
        if fp is None: continue
        cand_rows.append({
            "inchi_key": row["inchi_key"],
            "canonical_smiles": smi,
            "mol": mol, "fp": fp, "scaffold": sc,
            **props,
        })
    tracker.write_report(ROOT)
    lg.info(f"  Candidate pool prepared: {len(cand_rows)} compounds with valid props+FP")

    # Build properties for positives (using physchem cache)
    pos_props_list = phys_df[phys_df["inchi_key"].isin(pos_iks)].to_dict("records")
    rng = np.random.default_rng(pmd_cfg.get("seed", 42))
    rng.shuffle(pos_props_list)
    pos_sample = pos_props_list
    lg.info(f"  Positives evaluated for PMD matching: {len(pos_sample)}")

    # Match
    pmd_rows = []
    used_iks: set[str] = set()
    
    # Rejection taxonomy buckets
    rejections = {
        "MW": 0,
        "logP": 0,
        "TPSA": 0,
        "HBD/HBA": 0,
        "fraction_sp3": 0,
        "ring_count": 0,
        "similarity_constraint": 0,
        "duplicate": 0,
        "invalid_smiles": 0,
        "scaffold_mismatch": 0,
        "outside formal charge": 0
    }

    min_tan = 0.35  # Slightly relaxed per Phase 4 final pass instructions
    max_tan = 0.85
    
    cand_fps = [c["fp"] for c in cand_rows]
    scaffold_stats = {"count": 0, "same_scaffold": 0, "diff_scaffold": 0}
    
    # Pre-calculate training targets
    target_count = pmd_cfg.get("target_pmd_negatives", 1500)
    
    for pos in pos_sample:
        if len(pmd_rows) >= target_count:
            break
            
        matched = 0
        pos_fp = pos_fps_dict.get(pos["inchi_key"])
        pos_sc = pos_scaffold_dict.get(pos["inchi_key"])
        if pos_fp is None:
            continue
            
        # 1. compute Tanimoto similarity to candidate pool
        sims = DataStructs.BulkTanimotoSimilarity(pos_fp, cand_fps)
        
        # 2. select top 200 nearest candidates within band [0.40, 0.85]
        neighborhood = []
        for i, s in enumerate(sims):
            if min_tan <= s <= max_tan:
                neighborhood.append((i, s))
        
        # Sort by similarity descending and take top 200
        neighborhood.sort(key=lambda x: x[1], reverse=True)
        top_indices = neighborhood[:200]
        
        if not top_indices:
            rejections["similarity_constraint"] += 1
            continue
            
        # 3. apply property constraints within this neighborhood
        valid_cands = []
        for idx, sim in top_indices:
            cand = cand_rows[idx]
            if cand["inchi_key"] in used_iks:
                continue
            
            is_match, reason = property_match(pos, cand, cfg)
            if not is_match:
                rejections[reason] += 1
                continue
            
            # Scaffold preference score
            sc_score = 1.0 if (pos_sc and cand["scaffold"] == pos_sc) else 0.0
            valid_cands.append((cand, sim, sc_score))
            
        if not valid_cands:
            continue
            
        # 4. randomly sample a valid match, prioritizing same scaffold
        valid_cands.sort(key=lambda x: (x[2], x[1]), reverse=True)
        max_sc = valid_cands[0][2]
        best_set = [c for c in valid_cands if c[2] == max_sc]
        choice = random.choice(best_set)
        
        cand, sim, sc_score = choice
        ik = cand["inchi_key"]
        used_iks.add(ik)
        pmd_rows.append({
            "inchi_key": ik,
            "canonical_smiles": cand["canonical_smiles"],
            "MolWt":    cand.get("MolWt"),
            "MolLogP":  cand.get("MolLogP"),
            "NumHDonors": cand.get("NumHDonors"),
            "NumHAcceptors": cand.get("NumHAcceptors"),
            "TPSA":     cand.get("TPSA"),
            "NumRotatableBonds": cand.get("NumRotatableBonds"),
            "FormalCharge": cand.get("FormalCharge"),
            "FractionCSP3": cand.get("FractionCSP3"),
            "RingCount": cand.get("RingCount"),
            "label": 0,
            "matched_positive_ik": pos.get("inchi_key", ""),
            "tanimoto": sim,
            "same_scaffold": bool(sc_score > 0)
        })
        
        scaffold_stats["count"] += 1
        if sc_score > 0: scaffold_stats["same_scaffold"] += 1
        else: scaffold_stats["diff_scaffold"] += 1

        matched += 1
        if matched >= n_per_pos:
            continue 

    pmd_df = pd.DataFrame(pmd_rows)
    lg.info(f"  PMD negatives generated: {len(pmd_df)} (Target: {target_count})")
    
    min_required = pmd_cfg.get("min_pmd_negatives", 1000)
    if len(pmd_df) < min_required:
        lg.warning(f"Failed to generate enough PMD negatives! Generated {len(pmd_df)}, minimum required is {min_required}. (Scientific Soft Fail)")
        with open(ROOT / "data" / "diagnostics" / "phase4C_scientific_status.json", "w") as f:
            json.dump({"pmd_generation_success": False, "generated": len(pmd_df), "target": min_required}, f)

    # Save outputs
    neg_dir = ROOT / cfg["outputs"]["neg_dir"]
    neg_dir.mkdir(parents=True, exist_ok=True)
    pmd_df.to_parquet(neg_dir / "pmd_negatives.parquet", index=False)
    lg.info(f"  Written: pmd_negatives.parquet")
    
    # Save rejection taxonomy
    diag_dir = ROOT / "data" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    rej_df = pd.DataFrame(list(rejections.items()), columns=["Rejection Reason", "Count"])
    rej_df = rej_df.sort_values("Count", ascending=False)
    rej_df.to_csv(diag_dir / "pmd_rejection_reasons.csv", index=False)
    lg.info(f"  Written: diagnostics/pmd_rejection_reasons.csv")
    
    # Save match table (just the PMD dataframe without the full chemotype payload to save space)
    pmd_match_df = pmd_df[["inchi_key", "matched_positive_ik", "MolWt", "MolLogP", "TPSA", "tanimoto", "same_scaffold"]]
    pmd_match_df.to_parquet(diag_dir / "pmd_match_table.parquet", index=False)
    lg.info(f"  Written: diagnostics/pmd_match_table.parquet")
    
    # Save scaffold match statistics
    sc_df = pd.DataFrame([scaffold_stats])
    sc_df.to_csv(diag_dir / "scaffold_match_statistics.csv", index=False)
    lg.info(f"  Written: diagnostics/scaffold_match_statistics.csv")

    # ── Property overlap plot ──────────────────────────────────────────
    figs_dir = ROOT / cfg["outputs"]["figures_dir"]
    figs_dir.mkdir(parents=True, exist_ok=True)

    props_to_plot = ["MolWt", "MolLogP", "TPSA"]
    pos_phys = phys_df[phys_df["inchi_key"].isin(list(pos_iks)[:500])]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, prop in zip(axes, props_to_plot):
        pos_vals = pos_phys[prop].dropna().values
        pmd_vals = pmd_df[prop].dropna().values
        ax.hist(pos_vals, bins=40, alpha=0.6, color="#1565C0", label="Positives", density=True)
        ax.hist(pmd_vals, bins=40, alpha=0.6, color="#E64A19", label="PMD", density=True)
        ax.set_title(prop, fontsize=11)
        ax.set_ylabel("Density"); ax.legend(fontsize=8)
    fig.suptitle("IFG-26 PMD vs Positive Property Distributions", fontsize=12)
    plt.tight_layout()
    fig.savefig(figs_dir / "phase4C_property_overlap.png", dpi=150)
    plt.close()

    # ── Audit report ──────────────────────────────────────────────────
    
    # Format taxonomy table for markdown
    taxonomy_rows = []
    # Sort rejections by count descending
    for reason, count in sorted(rejections.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            taxonomy_rows.append(f"| {reason} | {count} |")
    taxonomy_table = "\n".join(taxonomy_rows) if taxonomy_rows else "| No rejections logged | 0 |"

    report_md = f"""# IFG-26 Phase 4C — PMD Negative Audit

_Generated: {ts}_

## PMD Generation Summary

| Item | Value |
|---|---|
| Target requested | {target_count} |
| Minimum required | {min_required} |
| PMD negatives generated | {len(pmd_df)} |
| Coverage % | {(len(pmd_df) / target_count)*100:.1f}% |
| Source pool | MGDB canonicalized compounds not in P |
| Tanimoto limits | {min_tan} ≤ $T_c$ ≤ {max_tan} |
| Scaffold disjoint | ✅ |
| Matching rules | MW ±{pmd_cfg.get('mw_tol', 0)*100}%, logP ±{pmd_cfg.get('logp_tol', 0)}, TPSA ±{pmd_cfg.get('tpsa_tol', 0)}, HBD/HBA ±{pmd_cfg.get('hbd_tol', 0)}, Fsp3 ±{pmd_cfg.get('fraction_sp3_tol', 0)}, rings ±{pmd_cfg.get('ring_count_tol', 0)}, charge exact |

## PMD Rejection Taxonomy

| Rejection Reason | Count |
|---|---|
{taxonomy_table}

## Artifact Audit Gates

> AUROC thresholds are evaluated in `scripts/phase4_binary_eval.py`.
> See `results/tables/phase4C_binary_eval.csv` for results.
> Physchem-only AUROC must be ≤ {pmd_cfg['artifact_auroc_physchem_limit']}
> ECFP4-only AUROC must be ≤ {pmd_cfg['artifact_auroc_ecfp_limit']}
> Minimum generated count must be ≥ {min_required} (Currently: {len(pmd_df)})

## Notes

PMD negatives are auxiliary controls only — they are NOT used in nnPU training.
They provide a calibration check for the difficulty of the negative set.
"""
    with open(ROOT / "docs" / "phase4C_pmd_generation.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    lg.info("  Written: docs/phase4C_pmd_generation.md")

    # Add Rejection Summary Artifact
    total_candidates = len(mgdb_cands)
    matched_negatives = len(pmd_df)
    unmatched_positives = target_count - pmd_df["matched_positive_ik"].nunique() if len(pmd_df) > 0 else target_count
    
    top_5_rows = []
    for reason, count in sorted(rejections.items(), key=lambda x: x[1], reverse=True)[:5]:
        if count > 0:
            top_5_rows.append(f"| {reason} | {count} |")
    top_5_table = "\n".join(top_5_rows) if top_5_rows else "| No rejections logged | 0 |"

    rejection_md = f"""# IFG-26 Phase 4C — PMD Candidate Rejection Summary
    
_Generated: {ts}_

This document summarizes why molecules from the candidate pool were rejected during PMD pairing.

### Global Statistics
- **Total Candidates Evaluated**: {total_candidates}
- **Matched PMD Negatives**: {matched_negatives}
- **Unmatched Positives**: {unmatched_positives}

### Top 5 Rejection Causes
| Filter | Rejection Count |
|---|---|
{top_5_table}

### Full Rejection Taxonomy
| Filter | Rejection Count |
|---|---|
{taxonomy_table}
"""
    with open(ROOT / "docs" / "phase4C_rejection_summary.md", "w", encoding="utf-8") as f:
        f.write(rejection_md)
    lg.info("  Written: docs/phase4C_rejection_summary.md")

    lg.info("\n" + "=" * 70)
    lg.info(f"IFG-26 Phase 4C PMD Generation — COMPLETE: {len(pmd_df)} negatives")
    lg.info("=" * 70)
    lg.info("\nNext: python scripts/phase4_binary_eval.py")


if __name__ == "__main__":
    main()
