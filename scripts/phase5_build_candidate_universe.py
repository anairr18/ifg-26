"""
phase5_build_candidate_universe.py
====================================
IFG-26 Phase 5 — Build Expanded Negative Candidate Universe.

Aggregates all non-positive ligands from MGDB, MGTbind, and internal
Phase 4 cleaning pools into a large deduplicated parquet.

Outputs:
    data/phase5_candidate_universe.parquet
    data/diagnostics/phase5_candidate_universe_stats.json
    data/diagnostics/mgtbind_parse_failures.csv
    docs/phase5_candidate_universe.md
    docs/phase5_candidate_universe_debug.md

Usage:
    python scripts/phase5_build_candidate_universe.py [--config path] [--resume]
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

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

import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from molecule_tracker import MoleculeTracker

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase4_default.yaml"


def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5_candidate_universe")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(log_dir / "phase5_candidate_universe.log", encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg


def compute_props(mol):
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
        return {}


def safe_inchikey(mol) -> str | None:
    try:
        ik = Chem.MolToInchiKey(mol)
        return ik if ik else None
    except Exception:
        return None


def load_source_with_debug(
    path: Path,
    smiles_col: str,
    id_col: str,
    source_name: str,
    tracker: MoleculeTracker,
    positive_iks: set,
    positive_smiles: set,
    lg,
    canonicalized_ok_col: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Load a CSV, parse molecules, compute props, skip positives.
    Returns (records, failure_rows) for detailed diagnosis.

    Staged rejection counters:
        not_found / empty_smiles / canonicalized_fail / overlap_ik /
        overlap_smiles / parse_fail / no_inchikey / no_props / accepted
    """
    failures = []
    records = []

    if not path.exists():
        lg.warning(f"  [{source_name}] File not found: {path}")
        return records, failures

    df = pd.read_csv(path, low_memory=False)
    lg.info(f"  [{source_name}] Raw rows: {len(df)}")
    lg.info(f"  [{source_name}] Columns: {df.columns.tolist()}")

    # Stage counts
    counts = defaultdict(int)

    # Sample first 10 non-null SMILES for diagnostics
    sample_smiles = df[smiles_col].dropna().head(10).tolist() if smiles_col in df.columns else []
    lg.info(f"  [{source_name}] Sample SMILES (first 10 non-null): {sample_smiles[:3]}")

    for i, row in df.iterrows():
        # --- Stage 1: Extract SMILES ----------------------------------------
        if smiles_col not in row or pd.isna(row[smiles_col]):
            counts["empty_smiles"] += 1
            failures.append({
                "source": source_name, "row_index": i,
                "raw_smiles": "", "stage": "empty_smiles",
                "id_col": str(row.get(id_col, ""))
            })
            continue
        smi = str(row[smiles_col]).strip()
        if not smi or smi == "nan":
            counts["empty_smiles"] += 1
            failures.append({
                "source": source_name, "row_index": i,
                "raw_smiles": smi, "stage": "empty_smiles",
                "id_col": str(row.get(id_col, ""))
            })
            continue

        # --- Stage 2: canonicalized_ok filter (if column exists) ------------
        if canonicalized_ok_col and canonicalized_ok_col in df.columns:
            ok_val = row.get(canonicalized_ok_col)
            # True means successfully canonicalized
            if str(ok_val).lower() not in ("true", "1", "yes"):
                counts["canonicalized_fail"] += 1
                failures.append({
                    "source": source_name, "row_index": i,
                    "raw_smiles": smi, "stage": "canonicalized_fail",
                    "id_col": str(row.get(id_col, ""))
                })
                continue

        # --- Stage 3: Extract ID --------------------------------------------
        ik_raw = str(row.get(id_col, "")).strip()

        # --- Stage 4: Positive overlap by InChIKey --------------------------
        if ik_raw and ik_raw != "nan" and ik_raw in positive_iks:
            counts["overlap_ik"] += 1
            continue

        # --- Stage 5: Positive overlap by SMILES (canonical) ----------------
        if smi in positive_smiles:
            counts["overlap_smiles"] += 1
            continue

        # --- Stage 6: RDKit parse -------------------------------------------
        mol = tracker.parse(smi, source_file=str(path.name),
                            stage=f"load_{source_name}", record_id=ik_raw or str(i))
        if mol is None:
            counts["parse_fail"] += 1
            failures.append({
                "source": source_name, "row_index": i,
                "raw_smiles": smi, "stage": "parse_fail",
                "id_col": ik_raw
            })
            continue

        # --- Stage 7: Compute InChIKey from parsed mol ----------------------
        inchi_key = safe_inchikey(mol)
        if not inchi_key:
            counts["no_inchikey"] += 1
            failures.append({
                "source": source_name, "row_index": i,
                "raw_smiles": smi, "stage": "no_inchikey",
                "id_col": ik_raw
            })
            continue

        # Double-check against positive IKs using the computed key
        if inchi_key in positive_iks:
            counts["overlap_ik"] += 1
            continue

        # --- Stage 8: Compute properties ------------------------------------
        props = compute_props(mol)
        if not props:
            counts["no_props"] += 1
            failures.append({
                "source": source_name, "row_index": i,
                "raw_smiles": smi, "stage": "no_props",
                "id_col": ik_raw
            })
            continue

        counts["accepted"] += 1
        records.append({
            "inchi_key": inchi_key,
            "smiles": Chem.MolToSmiles(mol),
            "source": source_name,
            **props,
        })

    lg.info(f"  [{source_name}] Stage breakdown:")
    for stage, cnt in counts.items():
        lg.info(f"    {stage}: {cnt}")
    lg.info(f"  [{source_name}] Accepted: {counts['accepted']}")

    return records, failures


def build_positive_smiles_set(split_paths: list, lg) -> tuple[set, set]:
    """
    Try to build a set of positive SMILES and InChIKeys from split files.
    Handles the case where no InChIKey column exists (new split format).
    """
    positive_iks = set()
    positive_smiles = set()

    for sp in split_paths:
        if not sp.exists():
            continue
        try:
            df = pd.read_csv(sp, low_memory=False)
            # InChIKey columns
            for col in ["ligand_inchikey", "compound_inchi_key", "inchi_key"]:
                if col in df.columns:
                    positive_iks.update(df[col].dropna().str.strip().tolist())
            # SMILES columns
            for col in ["canonical_smiles", "canonical_isomeric_smiles", "smiles"]:
                if col in df.columns:
                    positive_smiles.update(df[col].dropna().str.strip().tolist())
        except Exception as e:
            lg.warning(f"Failed to read {sp}: {e}")

    # Also try to compute InChIKeys from positive SMILES
    computed = 0
    for smi in list(positive_smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                ik = Chem.MolToInchiKey(mol)
                if ik:
                    positive_iks.add(ik)
                    computed += 1
        except Exception:
            pass

    lg.info(f"Positive IK from columns: {len(positive_iks) - computed}")
    lg.info(f"Positive IK computed from SMILES: {computed}")
    lg.info(f"Total positive InChIKeys to exclude: {len(positive_iks)}")
    lg.info(f"Total positive SMILES to exclude: {len(positive_smiles)}")
    return positive_iks, positive_smiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true", help="Skip if output already exists")
    args = parser.parse_args()

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    out_path = ROOT / "data" / "phase5_candidate_universe.parquet"

    if args.resume and out_path.exists():
        lg.info("phase5_candidate_universe.parquet already exists — skipping (--resume active).")
        return

    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 5 — Candidate Universe Build  {ts}")
    lg.info("=" * 70)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── Load positive exclusion sets ─────────────────────────────────────
    curated = ROOT / "data" / "curated" / "phase1"
    all_splits = [
        ROOT / "dataset/phase2/train_scaffold.csv",
        ROOT / "dataset/phase2/val_scaffold.csv",
        ROOT / "dataset/phase2/test_scaffold.csv",
        ROOT / "dataset/phase2/train_source.csv",
        ROOT / "dataset/phase2/val_source.csv",
        ROOT / "dataset/phase2/test_source.csv",
        curated / "ifg26_training_pairs.csv",
    ]
    positive_iks, positive_smiles = build_positive_smiles_set(all_splits, lg)

    tracker = MoleculeTracker(name="phase5_universe")
    all_records = []
    all_failures = []

    # ── Source 1: MGDB ────────────────────────────────────────────────────
    mgdb_path = curated / "mgdb_compounds_canonicalized.csv"
    recs, fails = load_source_with_debug(
        mgdb_path, "canonical_smiles", "inchi_key", "MGDB",
        tracker, positive_iks, positive_smiles, lg,
        canonicalized_ok_col="canonicalized_ok",
    )
    all_records.extend(recs)
    all_failures.extend(fails)

    # ── Source 2: MGTbind ─────────────────────────────────────────────────
    mgtbind_path = curated / "mgtbind_compounds_canonicalized.csv"
    recs, fails = load_source_with_debug(
        mgtbind_path, "canonical_smiles", "inchi_key", "MGTbind",
        tracker, positive_iks, positive_smiles, lg,
        canonicalized_ok_col="canonicalized_ok",
    )
    all_records.extend(recs)
    all_failures.extend(fails)

    # ── Source 3: Internal phase4 residuals ───────────────────────────────
    chem_fail_path = curated / "chem_failures.csv"
    if chem_fail_path.exists():
        df_cf = pd.read_csv(chem_fail_path, low_memory=False)
        for _, row in df_cf.iterrows():
            smi = str(row.get("smiles", row.get("canonical_smiles", ""))).strip()
            ik = str(row.get("inchi_key", "")).strip()
            if not smi or smi in ("nan", "") or smi in positive_smiles:
                continue
            if ik and ik != "nan" and ik in positive_iks:
                continue
            mol = tracker.parse(smi, source_file="chem_failures.csv",
                                stage="internal_residuals", record_id=ik or "?")
            if mol is None:
                continue
            inchi_key = safe_inchikey(mol)
            if not inchi_key or inchi_key in positive_iks:
                continue
            props = compute_props(mol)
            if props:
                all_records.append({
                    "inchi_key": inchi_key,
                    "smiles": Chem.MolToSmiles(mol),
                    "source": "internal_residuals",
                    **props,
                })
        lg.info("  [internal_residuals] Parsed and appended")
    else:
        lg.info("  [internal_residuals] chem_failures.csv not found — skipping")

    lg.info(f"Total records before dedup: {len(all_records)}")

    # ── Deduplicate ────────────────────────────────────────────────────────
    universe_df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    if not universe_df.empty:
        universe_df = universe_df.drop_duplicates(subset=["inchi_key"]).reset_index(drop=True)
    lg.info(f"Unique InChIKeys after dedup: {len(universe_df)}")

    # ── Write outputs ─────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    universe_df.to_parquet(out_path, index=False)
    lg.info(f"Written: {out_path.relative_to(ROOT)}")

    tracker.write_report(ROOT)

    diag_dir = ROOT / "data" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Save failure CSV
    if all_failures:
        fail_df = pd.DataFrame(all_failures)
        fail_path = diag_dir / "mgtbind_parse_failures.csv"
        # Write for MGTbind specifically
        mgtbind_fails = fail_df[fail_df["source"] == "MGTbind"] if "source" in fail_df.columns else fail_df
        mgtbind_fails.to_csv(fail_path, index=False)
        lg.info(f"Written: {fail_path.relative_to(ROOT)}")
        # Also write all failures
        fail_df.to_csv(diag_dir / "phase5_universe_all_failures.csv", index=False)

    # Stage breakdown for debug report
    stage_counts: dict[str, dict[str, int]] = {}
    if all_failures:
        fd = pd.DataFrame(all_failures)
        for src in fd["source"].unique():
            stage_counts[src] = fd[fd["source"] == src]["stage"].value_counts().to_dict()

    stats = {
        "timestamp": ts,
        "total_molecules": int(len(universe_df)),
        "unique_inchikeys": int(universe_df["inchi_key"].nunique()) if not universe_df.empty else 0,
        "parsed_ok": int(tracker.stats.get("parsed_ok", 0)),
        "total_failures": len(all_failures),
        "positive_iks_excluded": len(positive_iks),
        "positive_smiles_excluded": len(positive_smiles),
        "source_breakdown": universe_df["source"].value_counts().to_dict() if not universe_df.empty else {},
        "stage_rejection_counts": stage_counts,
    }
    stats_path = diag_dir / "phase5_candidate_universe_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    lg.info(f"Written: {stats_path.relative_to(ROOT)}")

    # ── Debug doc ─────────────────────────────────────────────────────────
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    stage_rows = ""
    for src, stages in stage_counts.items():
        for stage, cnt in stages.items():
            stage_rows += f"| {src} | {stage} | {cnt} |\n"

    src_accepted_rows = "\n".join(
        [f"| {s} | {c} |" for s, c in stats["source_breakdown"].items()]
    )

    debug_md = f"""# IFG-26 Phase 5 — Candidate Universe Debug Report

_Generated: {ts}_

## Summary

| Metric | Value |
|---|---|
| Total unique candidates (accepted) | {stats['total_molecules']} |
| Total parse failures logged | {stats['total_failures']} |
| Positive InChIKeys excluded | {stats['positive_iks_excluded']} |
| Positive SMILES excluded | {stats['positive_smiles_excluded']} |

## Accepted By Source

| Source | Count |
|---|---|
{src_accepted_rows if src_accepted_rows else "| (none) | 0 |"}

## Stage Rejection Breakdown

| Source | Stage | Count |
|---|---|---|
{stage_rows if stage_rows else "| (none) | (none) | 0 |"}

## Root Cause Analysis

### MGTbind parsed OK = 0

If MGTbind shows 0 accepted rows, check `stage_rejection_counts` in
`data/diagnostics/phase5_candidate_universe_stats.json`. Primary hypothesis:

1. **All MGTbind compounds overlap with positives** (`overlap_ik` or `overlap_smiles` dominant)
   - This is correct scientific behavior: MGTbind is a molecular glue binding database.
     Its compounds ARE the positives.
   
2. **canonicalized_ok filter too strict** (`canonicalized_fail` dominant)
   - If `canonicalized_ok == False` for most rows, the filter is excluding valid SMILES.
   - Fix: remove the `canonicalized_ok` filter and let RDKit decide independently.

3. **Parse failures** (`parse_fail` dominant)
   - Check `data/diagnostics/mgtbind_parse_failures.csv` for raw SMILES.

### Universe size < 3000

If the universe is too small for PMD-v2 matching, the recommended fix is to:
1. Not exclude MGTbind compounds by positive-overlap (since they ARE positives and won't
   appear as false negatives in the test set)
2. Or supplement with a public ChEMBL/ZINC fragment set

See `docs/phase5_scientific_interpretation.md` for scientific context.
"""
    with open(docs_dir / "phase5_candidate_universe_debug.md", "w", encoding="utf-8") as f:
        f.write(debug_md)
    lg.info("Written: docs/phase5_candidate_universe_debug.md")

    # ── Main doc ──────────────────────────────────────────────────────────
    doc_md = f"""# IFG-26 Phase 5 — Candidate Universe

_Generated: {ts}_

## Summary

| Metric | Value |
|---|---|
| Total molecules | {stats['total_molecules']} |
| Unique InChIKeys | {stats['unique_inchikeys']} |

## Source Breakdown

| Source | Count |
|---|---|
{src_accepted_rows if src_accepted_rows else "| (none) | 0 |"}

## Notes

This universe aggregates non-positive compounds from MGDB, MGTbind, and internal residuals.
See `docs/phase5_candidate_universe_debug.md` for a detailed rejection stage breakdown.
"""
    with open(docs_dir / "phase5_candidate_universe.md", "w", encoding="utf-8") as f:
        f.write(doc_md)
    lg.info("Written: docs/phase5_candidate_universe.md")

    lg.info("=" * 70)
    lg.info(f"Phase 5 Candidate Universe — COMPLETE: {len(universe_df)} compounds")
    lg.info("=" * 70)


if __name__ == "__main__":
    main()
