"""
phase4E_expand_decoy_pool.py
============================
IFG-26 Phase 4E — Decoy Pool Expansion.

Loads MGDB and MGTbind compounds, computes RDKit properties, filters against P,
and generates a combined Decoy pool.
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.append(str(Path(__file__).resolve().parent))
from molecule_tracker import MoleculeTracker

rdBase.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

def setup_logging(name="phase4E_decoy_expansion"):
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt); lg.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); lg.addHandler(sh)
    return lg

def abort(lg, msg):
    lg.error(f"[ABORTED] {msg}")
    sys.exit(1)

def compute_props(smiles, tracker, source_file, stage, record_id):
    try:
        mol = tracker.parse(smiles, source_file=source_file, stage=stage, record_id=record_id)
        if mol is None: return None, None, None
        ik = Chem.MolToInchiKey(mol)
        murcko = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        desc = {
            'MW': Descriptors.MolWt(mol),
            'cLogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'RotB': Descriptors.NumRotatableBonds(mol),
            'FormalCharge': Chem.GetFormalCharge(mol)
        }
        return ik, murcko, json.dumps(desc)
    except Exception:
        return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run in smoke-test mode (only validate loading and first-stage filtering)")
    args = parser.parse_args()

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 4E — Decoy Pool Expansion  {ts}")
    lg.info("=" * 70)

    # ── Input Definitions ─────────────────────────────────────────────
    # Depending on where the pipeline drops these Phase 0/1 files:
    mgtbind_path = ROOT / "data" / "raw" / "mgtbind" / "mgtbind_compounds.csv"
    mgdb_path = ROOT / "data" / "curated" / "phase1" / "mgdb_compounds_canonicalized.csv"
    # Fallback to pure raw if curated doesn't exist
    if not mgdb_path.exists():
        mgdb_path = ROOT / "data" / "raw" / "mgdb" / "mgdb_compounds.csv"

    p_pool_path = ROOT / "data" / "pu" / "pool_P_scaffold.parquet"

    # ── Load Raw Records ──────────────────────────────────────────────
    records = []
    telemetry = {"discovered_files": [], "rows_loaded": {}, "rows_filtered": {}}

    lg.info("Scanning for raw dataset inputs...")

    if mgtbind_path.exists():
        telemetry["discovered_files"].append(str(mgtbind_path))
        mgt = pd.read_csv(mgtbind_path, low_memory=False)
        telemetry["rows_loaded"]["mgtbind"] = len(mgt)
        if 'smiles' in mgt.columns:
            for sm in mgt['smiles'].dropna().unique():
                records.append({'smiles': sm, 'source_tag': 'mgtbind'})
        elif 'canonical_smiles' in mgt.columns:
             for sm in mgt['canonical_smiles'].dropna().unique():
                records.append({'smiles': sm, 'source_tag': 'mgtbind'})
        lg.info(f"  Loaded MGTbind: {len(mgt)} raw rows")
    else:
        lg.warning(f"  Missing MGTbind path: {mgtbind_path}")

    if mgdb_path.exists():
        telemetry["discovered_files"].append(str(mgdb_path))
        try:
            mgdb = pd.read_csv(mgdb_path, low_memory=False)
            if 'canonical_smiles' not in mgdb.columns and 'Smiles' not in mgdb.columns:
                mgdb = pd.read_csv(mgdb_path, skiprows=1, low_memory=False)
        except Exception as e:
            abort(lg, f"Failed to read MGDB file: {e}")
            
        telemetry["rows_loaded"]["mgdb"] = len(mgdb)
        if 'canonical_smiles' in mgdb.columns:
            for sm in mgdb['canonical_smiles'].dropna().unique():
                records.append({'smiles': sm, 'source_tag': 'mgdb'})
        elif 'Smiles' in mgdb.columns:
            for sm in mgdb['Smiles'].dropna().unique():
                records.append({'smiles': sm, 'source_tag': 'mgdb'})
        lg.info(f"  Loaded MGDB: {len(mgdb)} raw rows")
    else:
        lg.warning(f"  Missing MGDB path: {mgdb_path}")

    if not records:
        abort(lg, f"0 raw records loaded! Checked paths:\n - {mgtbind_path}\n - {mgdb_path}")

    lg.info(f"Total raw SMILES extracted: {len(records)}")
    telemetry["rows_filtered"]["1_unique_raw_smiles"] = len(records)

    if args.smoke:
        lg.info("SMOKE TEST MODE: Successfully validated raw loading. Halting execution.")
        return

    # ── Compute RDKit Props ───────────────────────────────────────────
    lg.info("Computing RDKit properties and generating InChIKeys...")
    processed = []
    failed_rdkit = 0
    tracker = MoleculeTracker(name="phase4E_expand_decoy_pool")

    for i, r in enumerate(records):
        ik, murcko, desc = compute_props(r['smiles'], tracker, source_file=r['source_tag'], stage="decoy_expansion", record_id=str(i))
        if ik:
            processed.append({
                'ligand_id': f"{r['source_tag']}_{ik[:8]}",
                'smiles': r['smiles'],
                'inchikey': ik,
                'murcko': murcko,
                'rdkit_desc': desc,
                'source_tag': r['source_tag']
            })
        else:
            failed_rdkit += 1

    telemetry["rows_filtered"]["2_failed_rdkit_parse"] = failed_rdkit
    
    fail_csv, stat_json = tracker.write_report(ROOT)
    lg.info(f"MoleculeTracker logged {tracker.stats['total_raw']} raw, {tracker.stats['parsed_ok']} OK, {tracker.stats['parse_failure']} Parse Fails, {tracker.stats['valence_error']} Valence Errors, {tracker.stats['sanitize_failure']} Sanitize Fails")
    lg.info(f"Wrote invalid records tracking to {fail_csv}")
    
    df = pd.DataFrame(processed)
    if df.empty:
        abort(lg, "0 unique decoys generated after RDKit processing.")
        
    df = df.drop_duplicates(subset=['inchikey'])
    telemetry["rows_filtered"]["3_unique_inchikeys"] = len(df)
    lg.info(f"Unique valid compounds after RDKit: {len(df)}")

    # ── Filter against Positives ──────────────────────────────────────
    lg.info("Filtering out known Positives (P pool)...")
    if p_pool_path.exists():
        p_df = pd.read_parquet(p_pool_path)
        if 'ligand_inchikey' in p_df.columns:
            p_iks = set(p_df['ligand_inchikey'].dropna())
            overlap = df['inchikey'].isin(p_iks)
            telemetry["rows_filtered"]["4_removed_p_overlap"] = overlap.sum()
            df = df[~overlap]
        else:
            lg.warning("P pool missing 'ligand_inchikey' column; skipping strict deduplication.")
    else:
        lg.warning(f"P pool not found at {p_pool_path}; skipping positive filtering.")

    final_count = len(df)
    telemetry["rows_filtered"]["5_final_decoys"] = final_count
    
    if final_count == 0:
        abort(lg, "0 unique decoys generated after all filters (overlap removed all candidates).")

    # ── Save Outputs ──────────────────────────────────────────────────
    lg.info(f"Generating {final_count} final unique decoys.")
    
    neg_dir = ROOT / "data" / "negatives"
    neg_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(neg_dir / "decoy_pool.parquet", index=False)
    lg.info(f"Written: data/negatives/decoy_pool.parquet")

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    debug_md = f"""# Phase 4E Decoy Pool Expansion Debug log

_Generated: {ts}_

## Input Telemetry
**Discovered Files:**
{chr(10).join(f"- {f}" for f in telemetry['discovered_files'])}

**Raw Rows Loaded:**
{chr(10).join(f"- {k}: {v}" for k,v in telemetry['rows_loaded'].items())}

## Filtering Pipeline
| Stage | Count | Description |
|-------|-------|-------------|
| 1. Raw SMILES unique | {telemetry['rows_filtered'].get('1_unique_raw_smiles', 0)} | De-duplicated string-based SMILES array |
| 2. RDKit parse failures | {telemetry['rows_filtered'].get('2_failed_rdkit_parse', 0)} | SMILES that failed `MolFromSmiles` |
| 3. Unique InChIKeys | {telemetry['rows_filtered'].get('3_unique_inchikeys', 0)} | Valid objects deduped by Exact InChIKey |
| 4. Removed P overlap | {telemetry['rows_filtered'].get('4_removed_p_overlap', 0)} | Filtered against `{p_pool_path.name}` |
| **5. Final Decoys** | **{telemetry['rows_filtered'].get('5_final_decoys', 0)}** | **Total written to `decoy_pool.parquet`** |

## Notes
- Smoke test mode: {'Enabled' if args.smoke else 'Disabled'}
"""
    with open(docs_dir / "phase4E_decoy_expansion_debug.md", 'w', encoding="utf-8") as f:
        f.write(debug_md)
    lg.info("Written: docs/phase4E_decoy_expansion_debug.md")

    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 4E Decoy Expansion — COMPLETE: {final_count} decoys")
    lg.info("=" * 70)

if __name__ == '__main__':
    main()
