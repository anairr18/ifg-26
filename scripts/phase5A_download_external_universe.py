"""
phase5A_download_external_universe.py
=======================================
IFG-26 Phase 5A — Download and Process External ChEMBL Universe.

Retrieves the ChEMBL 34 chemical representations, filters for drug-like
Lipinski-compliant molecules, and removes any overlap with IFG-26 positives.

Requirements:
    - ~100k-500k drug-like ligands
    - Salt removal
    - Lipinski-like filters
    - Zero overlap with positives

Outputs:
    data/external_chembl_universe.parquet
    data/diagnostics/external_universe_stats.json
    docs/phase5_external_universe_build.md
"""

import os
import sys
import gzip
import logging
import urllib.request
import warnings
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

# --- Environment Guards ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONNOUSERSITE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from molecule_tracker import MoleculeTracker

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

CHEMBL_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/chembl_34_chemreps.txt.gz"
LOCAL_GZ = ROOT / "data/raw/chembl_34_chemreps.txt.gz"

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5A_chembl")
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5A_chembl.log", encoding="utf-8"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg

def download_file(url, target, lg):
    if target.exists():
        lg.info(f"File already exists: {target}")
        return
    lg.info(f"Downloading from {url} (using urllib) ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=300) as response, open(target, "wb") as f:
            chunk_size = 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk: break
                f.write(chunk)
        lg.info("Download complete.")
    except Exception as e:
        lg.error(f"Download failed: {e}")
        raise

def is_drug_like(mol):
    """Lipinski-like filters."""
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        if mw < 500 and logp < 5 and hbd < 5 and hba < 10:
            return True, {"MolWt": mw, "MolLogP": logp, "NumHDonors": hbd, "NumHAcceptors": hba}
    except Exception: pass
    return False, {}

def get_positives(lg):
    pos_iks = set()
    pos_smiles = set()
    # Check common locations for curated positives
    paths = [
        ROOT / "dataset/phase1/mgdb_compounds_canonicalized.csv",
        ROOT / "dataset/phase1/mgtbind_compounds_canonicalized.csv",
        ROOT / "dataset/phase2/train_scaffold.csv",
        ROOT / "dataset/phase2/val_scaffold.csv",
        ROOT / "dataset/phase2/test_scaffold.csv"
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            for col in ["inchi_key", "ligand_inchikey", "compound_inchi_key"]:
                if col in df.columns: pos_iks.update(df[col].dropna().unique())
            for col in ["canonical_smiles", "smiles"]:
                if col in df.columns: pos_smiles.update(df[col].dropna().unique())
    lg.info(f"Loaded {len(pos_iks)} positive InChIKeys for exclusion.")
    return pos_iks, pos_smiles

def main():
    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    lg.info("Starting Phase 5A: ChEMBL External Universe Build")

    # 1. Download
    download_file(CHEMBL_URL, LOCAL_GZ, lg)

    # 2. Load Positives
    pos_iks, pos_smiles = get_positives(lg)

    # 3. Process
    tracker = MoleculeTracker(name="phase5A_chembl")
    records = []
    stats = {"total_raw": 0, "chembl_parse_fail": 0, "positive_overlap": 0, "not_drug_like": 0, "accepted": 0}

    lg.info("Processing ChEMBL file...")
    with gzip.open(LOCAL_GZ, "rt", encoding="utf-8") as f:
        # chemreps format: chembl_id \t smiles \t standard_inchi \t standard_inchi_key
        header = f.readline()
        for i, line in enumerate(f):
            stats["total_raw"] += 1
            if i % 100000 == 0: lg.info(f"  Processed {i} rows...")
            
            parts = line.strip().split("\t")
            if len(parts) < 4: continue
            cid, smi, inchi, ik = parts[0], parts[1], parts[2], parts[3]

            # Fast overlap check before RDKit
            if ik in pos_iks or smi in pos_smiles:
                stats["positive_overlap"] += 1
                continue

            # RDKit processing
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                stats["chembl_parse_fail"] += 1
                continue
            
            # Salt removal (keep largest fragment)
            if "." in smi:
                mol = Chem.MolFromSmiles(max(smi.split("."), key=len))
                if not mol: continue
            
            ok, props = is_drug_like(mol)
            if not ok:
                stats["not_drug_like"] += 1
                continue
            
            # Final check on canonicalized
            can_smi = Chem.MolToSmiles(mol, canonical=True)
            if can_smi in pos_smiles:
                stats["positive_overlap"] += 1
                continue

            stats["accepted"] += 1
            records.append({
                "inchi_key": ik,
                "smiles": can_smi,
                "chembl_id": cid,
                "source": "chembl_34",
                **props,
                "TPSA": rdMolDescriptors.CalcTPSA(mol),
                "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
                "RingCount": rdMolDescriptors.CalcNumRings(mol)
            })
            
            if stats["accepted"] >= 500000: # Target limit
                lg.info("Target limit (500k) reached.")
                break

    # 4. Save
    out_df = pd.DataFrame(records)
    out_path = ROOT / "data/external_chembl_universe.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    lg.info(f"Saved {len(out_df)} drug-like molecules to {out_path}")

    # 5. Report
    stats_path = ROOT / "data/diagnostics/external_universe_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        import json
        json.dump(stats, f, indent=2)

    doc_md = f"""# IFG-26 Phase 5A — External Universe Build (ChEMBL)

_Generated: {ts}_

## Core Metrics
| Metric | Value |
|---|---|
| Total ChEMBL entries processed | {stats['total_raw']} |
| Filtered (Drug-like only) | {stats['accepted']} |
| Removed (Overlap with positives) | {stats['positive_overlap']} |
| Dropped (Non-drug-like) | {stats['not_drug_like']} |

## QC Filters
- Salt removal: Largest fragment kept.
- Lipinski: MW < 500, LogP < 5, HBD < 5, HBA < 10.
- Overlap: Strictly excluded positives via InChIKey and SMILES.

## Output
`data/external_chembl_universe.parquet`
"""
    doc_path = ROOT / "docs/phase5_external_universe_build.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(doc_path, "w") as f:
        f.write(doc_md)
    lg.info(f"Build report saved to {doc_path}")

if __name__ == "__main__":
    main()
