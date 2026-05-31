"""
phase5_generate_pmd_v2.py
===========================
IFG-26 Phase 5 — Expanded Property-Matched Decoy Generator (PMD-v2).

Uses the expanded candidate universe from Phase 5 to generate strictly
property-matched decoys for each positive ligand. One PMD per positive,
chosen by minimizing a combined distance score.

PMD tolerances (v2):
    |ΔMW|   <= 7%      |ΔlogP|  <= 0.7
    |ΔTPSA| <= 10%     |ΔHBD|   <= 1
    |ΔHBA|  <= 1       |ΔFsp3|  <= 0.1
    |Δrings|<= 1
    0.20 <= ECFP4 Tanimoto <= 0.65

Score function:
    score = 0.7 * norm_property_distance + 0.3 * tanimoto_band_penalty

Outputs:
    data/phase5_pmd_v2_negatives.parquet
    data/diagnostics/phase5_pmd_v2_stats.json
    data/diagnostics/phase5_pmd_v2_rejections.csv
    docs/phase5_pmd_v2_generation.md

Usage:
    python scripts/phase5_generate_pmd_v2.py [--config path] [--resume]
"""

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from molecule_tracker import MoleculeTracker

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase4_default.yaml"

# PMD-v2 tolerances (stricter Tanimoto band than v1)
TOL = {
    "mw_frac":   0.07,   # ±7% MW
    "logp_abs":  0.70,
    "tpsa_frac": 0.10,   # ±10% TPSA
    "hbd_int":   1,
    "hba_int":   1,
    "fsp3_abs":  0.10,
    "ring_int":  1,
    "tan_min":   0.20,   # min Tanimoto similarity to positive
    "tan_max":   0.65,   # max Tanimoto similarity to positive
}


def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5_pmd_v2")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5_pmd_v2.log", encoding="utf-8"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg


def get_props(mol) -> dict | None:
    if mol is None:
        return None
    try:
        return {
            "MolWt": Descriptors.MolWt(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
            "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "FormalCharge": Chem.GetFormalCharge(mol),
        }
    except Exception:
        return None


def get_fp(mol):
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return None


def get_scaffold(mol) -> str | None:
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True) if core else None
    except Exception:
        return None


def property_match(pos_p: dict, cand_p: dict) -> tuple[bool, str]:
    """Return (pass, rejection_reason). 'pass' if all tolerances satisfied."""
    mw = pos_p["MolWt"]
    if mw <= 0:
        return False, "invalid_positive_mw"
    if abs(cand_p["MolWt"] - mw) > TOL["mw_frac"] * mw:
        return False, "MW"
    if abs(cand_p["MolLogP"] - pos_p["MolLogP"]) > TOL["logp_abs"]:
        return False, "logP"
    tpsa = pos_p["TPSA"]
    tpsa_tol = max(TOL["tpsa_frac"] * tpsa, 5.0)   # floor 5 Å²
    if abs(cand_p["TPSA"] - tpsa) > tpsa_tol:
        return False, "TPSA"
    if abs(cand_p["NumHDonors"] - pos_p["NumHDonors"]) > TOL["hbd_int"]:
        return False, "HBD"
    if abs(cand_p["NumHAcceptors"] - pos_p["NumHAcceptors"]) > TOL["hba_int"]:
        return False, "HBA"
    if abs(cand_p["FractionCSP3"] - pos_p["FractionCSP3"]) > TOL["fsp3_abs"]:
        return False, "fraction_sp3"
    if abs(cand_p["RingCount"] - pos_p["RingCount"]) > TOL["ring_int"]:
        return False, "ring_count"
    return True, "pass"


def norm_prop_distance(pos_p: dict, cand_p: dict) -> float:
    """Compute a normalized property distance score (lower is better match)."""
    mw = max(pos_p["MolWt"], 1.0)
    tpsa = max(pos_p["TPSA"], 1.0)
    parts = [
        abs(cand_p["MolWt"] - pos_p["MolWt"]) / (TOL["mw_frac"] * mw),
        abs(cand_p["MolLogP"] - pos_p["MolLogP"]) / TOL["logp_abs"],
        abs(cand_p["TPSA"] - pos_p["TPSA"]) / (TOL["tpsa_frac"] * tpsa),
        abs(cand_p["NumHDonors"] - pos_p["NumHDonors"]) / TOL["hbd_int"],
        abs(cand_p["NumHAcceptors"] - pos_p["NumHAcceptors"]) / TOL["hba_int"],
        abs(cand_p["FractionCSP3"] - pos_p["FractionCSP3"]) / TOL["fsp3_abs"],
        abs(cand_p["RingCount"] - pos_p["RingCount"]) / max(TOL["ring_int"], 1),
    ]
    return float(np.mean(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    out_path = ROOT / "data" / "phase5_pmd_v2_negatives.parquet"

    if args.resume and out_path.exists():
        lg.info("phase5_pmd_v2_negatives.parquet exists — skipping (--resume).")
        return

    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 5 — PMD-v2 Generation  {ts}")
    lg.info("=" * 70)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── Load candidate universe ──────────────────────────────────────────
    universe_path = ROOT / "data" / "phase5_candidate_universe.parquet"
    if not universe_path.exists():
        lg.error("phase5_candidate_universe.parquet not found. Run phase5_build_candidate_universe.py first.")
        sys.exit(1)
    universe_df = pd.read_parquet(universe_path)
    lg.info(f"Candidate universe loaded: {len(universe_df)}")

    # ── Load positives ────────────────────────────────────────────────────
    curated_p2 = ROOT / "data" / "curated" / "phase2"
    splits = [curated_p2 / "train_scaffold.csv", curated_p2 / "val_scaffold.csv", curated_p2 / "test_scaffold.csv"]
    pos_rows = []
    for sp in splits:
        if sp.exists():
            pos_rows.append(pd.read_csv(sp, low_memory=False))
    if not pos_rows:
        lg.error("No positive split CSVs found.")
        sys.exit(1)
    pos_df = pd.concat(pos_rows, ignore_index=True)

    # Detection of SMILES and InChIKey columns
    pos_smiles_col = next((c for c in ["canonical_smiles", "canonical_isomeric_smiles", "smiles"] if c in pos_df.columns), None)
    ik_col = next((c for c in ["ligand_inchikey", "compound_inchi_key", "inchi_key"] if c in pos_df.columns), None)

    if pos_smiles_col is None:
        lg.error(f"No SMILES column found in split CSVs. Cols: {pos_df.columns.tolist()}")
        sys.exit(1)

    lg.info(f"Raw positives loaded: {len(pos_df)}")

    # Ensure InChIKey exists for deduplication
    tracker = MoleculeTracker(name="phase5_pmd_v2")
    if ik_col is None:
        lg.info("No InChIKey column found. Computing InChIKeys for positives...")
        ik_list = []
        for smi in pos_df[pos_smiles_col].tolist():
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                ik = Chem.MolToInchiKey(mol)
                ik_list.append(ik if ik else "INVALID")
            else:
                ik_list.append("INVALID")
        pos_df["ligand_inchikey"] = ik_list
        ik_col = "ligand_inchikey"
    else:
        pos_df = pos_df.rename(columns={ik_col: "ligand_inchikey"})
        ik_col = "ligand_inchikey"

    # Deduplicate positives by InChIKey
    pos_df = pos_df[pos_df["ligand_inchikey"] != "INVALID"].drop_duplicates("ligand_inchikey").reset_index(drop=True)
    lg.info(f"Unique positive ligands after dedup: {len(pos_df)}")

    # ── Pre-compute candidate FPs and props ──────────────────────────────
    lg.info("Pre-computing candidate ECFP4 fingerprints...")
    cand_fps = []
    for i, row in universe_df.iterrows():
        smi = str(row.get("smiles", ""))
        mol = Chem.MolFromSmiles(smi)
        cand_fps.append(get_fp(mol))
    lg.info(f"Candidate FPs computed: {sum(fp is not None for fp in cand_fps)}")

    prop_cols = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
                 "FractionCSP3", "RingCount", "FormalCharge"]
    cand_props_list = universe_df[prop_cols].fillna(0).to_dict("records")

    # ── Match each positive ──────────────────────────────────────────────
    lg.info("Matching positives to candidates...")
    rejections = defaultdict(int)
    pmd_records = []
    already_used = set()  # one PMD per candidate
    pos_scaffolds = set()

    # Pre-collect all positive scaffolds for disjoint check
    for _, prow in pos_df.iterrows():
        smi = str(prow.get(pos_smiles_col, ""))
        mol = Chem.MolFromSmiles(smi)
        sc = get_scaffold(mol)
        if sc:
            pos_scaffolds.add(sc)
    lg.info(f"Positive scaffolds indexed: {len(pos_scaffolds)}")

    # Run matching
    for _, prow in pos_df.iterrows():
        smi = str(prow.get(pos_smiles_col, ""))
        pos_ik = str(prow.get("ligand_inchikey", ""))
        mol = tracker.parse(smi, source_file="splits", stage="pos_match", record_id=pos_ik)
        if mol is None:
            rejections["invalid_positive"] += 1
            continue
        pos_p = get_props(mol)
        if not pos_p:
            rejections["invalid_positive_props"] += 1
            continue
        pos_fp = get_fp(mol)
        if pos_fp is None:
            rejections["invalid_positive_fp"] += 1
            continue

        best_score = float("inf")
        best_candidate = None

        for idx, (cand_fp, cand_p) in enumerate(zip(cand_fps, cand_props_list)):
            cand_ik = universe_df.iloc[idx]["inchi_key"]
            if cand_ik in already_used:
                rejections["duplicate"] += 1
                continue
            if cand_fp is None:
                rejections["invalid_cand_fp"] += 1
                continue

            # 1. Property Box Match
            ok, reason = property_match(pos_p, cand_p)
            if not ok:
                rejections[reason] += 1
                continue

            # 2. Tanimoto Band Match
            tan = DataStructs.TanimotoSimilarity(pos_fp, cand_fp)
            if tan < TOL["tan_min"]:
                rejections["tanimoto_low"] += 1
                continue
            if tan > TOL["tan_max"]:
                rejections["tanimoto_high"] += 1
                continue

            # 3. Scaffold Disjoint Check
            # (Candidate core must not be in the set of ALL positive cores)
            cand_smi = str(universe_df.iloc[idx]["smiles"])
            c_mol = Chem.MolFromSmiles(cand_smi)
            c_sc = get_scaffold(c_mol)
            if c_sc and c_sc in pos_scaffolds:
                rejections["scaffold_overlap"] += 1
                continue

            # 4. Scoring
            p_dist = norm_prop_distance(pos_p, cand_p)
            tan_center = (TOL["tan_min"] + TOL["tan_max"]) / 2
            tan_pen = abs(tan - tan_center) / ((TOL["tan_max"] - TOL["tan_min"]) / 2)
            score = 0.7 * p_dist + 0.3 * tan_pen

            if score < best_score:
                best_score = score
                best_candidate = (idx, cand_ik, tan, score)

        if best_candidate:
            idx, cand_ik, tan, score = best_candidate
            already_used.add(cand_ik)
            cand_row = universe_df.iloc[idx]
            pmd_records.append({
                "inchi_key": cand_ik,
                "smiles": cand_row["smiles"],
                "source": cand_row.get("source", "unknown"),
                "matched_positive_ik": pos_ik,
                "tanimoto_to_pos": tan,
                "score": score,
                **{col: cand_row.get(col) for col in prop_cols},
            })

    lg.info(f"PMD-v2 negatives generated: {len(pmd_records)}")

    # ── Write outputs ────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pmd_df = pd.DataFrame(pmd_records)
    pmd_df.to_parquet(out_path, index=False)
    lg.info(f"Written: {out_path.relative_to(ROOT)}")

    tracker.write_report(ROOT)

    diag_dir = ROOT / "data" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Rejection Stats
    rej_df = pd.DataFrame(list(rejections.items()), columns=["reason", "count"])
    rej_df = rej_df.sort_values("count", ascending=False)
    rej_df.to_csv(diag_dir / "phase5_pmd_v2_rejections.csv", index=False)

    stats = {
        "timestamp": ts,
        "total_candidates": int(len(universe_df)),
        "positives_evaluated": int(len(pos_df)),
        "pmd_v2_generated": int(len(pmd_records)),
        "rejection_breakdown": dict(rejections),
    }
    with open(diag_dir / "phase5_pmd_v2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Docs ──────────────────────────────────────────────────────────────
    rej_rows = "\n".join([f"| {r['reason']} | {r['count']} |" for _, r in rej_df.iterrows()])
    doc_md = f"""# IFG-26 Phase 5 — PMD-v2 Generation

_Generated: {ts}_

| Metric | Value |
|---|---|
| Candidate universe | {len(universe_df)} |
| Positives evaluated | {len(pos_df)} |
| PMD-v2 generated | {len(pmd_records)} |

## Rejection Breakdown

| Reason | Count |
|---|---|
{rej_rows}

## Notes
Strict property matching logic:
- MW ±7%
- logP ±0.7
- TPSA ±10% (min 5)
- HBD/HBA ±1
- Fsp3 ±0.1
- Rings ±1
- Tanimoto Band: 0.20 - 0.65
- Scaffold Disjoint: Candidate scaffold must not exist in ANY positive scaffold set.
"""
    with open(ROOT / "docs" / "phase5_pmd_v2_generation.md", "w", encoding="utf-8") as f:
        f.write(doc_md)

    lg.info("DONE.")


if __name__ == "__main__":
    main()
