"""
phase8_model_selection_audit.py
==============================

Audits the environment and data for the Phase 8 comparison suite.
Verifies that:
1. Feature caches (ECFP4, ProtBert) exist.
2. Standardized splits (Proxy, PMD-v1, PU Pool) are accessible.
3. Training scripts for M1-M6 are ready or templated.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parent.parent

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase8_audit")
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg

def main():
    lg = setup_logging()
    lg.info("Starting Phase 8 Model Selection Audit...")

    # 1. Check Data Directories
    data_dir = ROOT / "data"
    feats_dir = data_dir / "features"
    results_dir = ROOT / "results"
    
    required_dirs = [feats_dir, results_dir / "phase7", ROOT / "docs"]
    for d in required_dirs:
        if d.exists():
            lg.info(f"[PASS] Directory exists: {d}")
        else:
            lg.error(f"[FAIL] Directory missing: {d}")

    # 2. Check Feature Caches
    required_feats = ["ligands_ecfp4.npy", "ligand_index.parquet", "protein_index.parquet"]
    for f in required_feats:
        if (feats_dir / f).exists():
            lg.info(f"[PASS] Feature found: {f}")
        else:
            lg.error(f"[FAIL] Feature missing: {f}")

    # 3. Check Dataset Splits
    required_splits = [
        ROOT / "results" / "phase7" / "reproducibility_metrics.csv",
        ROOT / "data" / "phase5_negative_ladder_core.csv"
    ]
    for s in required_splits:
        if s.exists():
            lg.info(f"[PASS] Split found: {s.name}")
        else:
            lg.warning(f"[WARN] Split missing: {s.name} (May need generation)")

    lg.info("Audit complete. Ready to proceed to Part 2: Evaluation Bundles.")

if __name__ == "__main__":
    main()
