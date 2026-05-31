"""
phase4_binary_eval.py
=====================
IFG-26 Phase 4C-ii — Auxiliary Binary Evaluation (PMD Artifact Audit).

Trains physchem-only and ECFP4-only classifiers on P vs PMD negatives
to verify that PMD negatives are not trivially separable.

Artifact audit gates:
    Physchem-only AUROC ≤ 0.70  (warn if exceeded → PMD too easy)
    ECFP4-only    AUROC ≤ 0.85  (warn if exceeded → tighten PMD constraints)

NOTE: This is an AUXILIARY sanity check only.
      Results MUST NOT be compared to nnPU headline metrics.
      AUROC is appropriate here because P vs PMD is binary, but
      this binary regime is explicitly labeled as "auxiliary".

Outputs:
    results/tables/phase4C_binary_eval.csv
    docs/phase4C_negative_audit.md  (appended)
"""

import json
import logging
import os
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

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase4_default.yaml"


def setup_logging(name="phase4_binary_eval"):
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


def cv_auroc(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[float, float]:
    """5-fold CV AUROC with LogisticRegression."""
    if len(np.unique(y)) < 2:
        return 0.5, 0.0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X, y):
        model = LogisticRegression(max_iter=1000, solver="lbfgs",
                                   random_state=seed, n_jobs=-1)
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[va])[:, 1]
        if len(np.unique(y[va])) > 1:
            scores.append(roc_auc_score(y[va], prob))
    return float(np.mean(scores)) if scores else 0.5, float(np.std(scores)) if scores else 0.0


def main():
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 4C-ii — Binary PMD Artifact Eval  {ts}")
    lg.info("=" * 70)

    pmd_cfg = cfg.get("pmd", {})
    physchem_limit = pmd_cfg.get("artifact_auroc_physchem_limit", 0.70)
    ecfp_limit     = pmd_cfg.get("artifact_auroc_ecfp_limit",     0.85)

    feats_dir = ROOT / "data" / "features"
    neg_dir   = ROOT / cfg["outputs"]["neg_dir"]

    pmd_df   = pd.read_parquet(neg_dir / "pmd_negatives.parquet")
    phys_df  = pd.read_parquet(feats_dir / "ligands_physchem.parquet")
    ecfp_mat = np.load(str(feats_dir / "ligands_ecfp4.npy"))
    lig_idx  = pd.read_parquet(feats_dir / "ligand_index.parquet")
    pairs    = pd.read_parquet(feats_dir / "scaffold_pairs_index.parquet")

    # Sample positives to match PMD size
    pos_iks = pd.Series(pairs["ligand_inchikey"].unique())
    if len(pos_iks) > len(pmd_df):
        pos_iks = pos_iks.sample(len(pmd_df), random_state=42)

    ik_to_row = lig_idx.set_index("inchi_key")["row_idx"].to_dict()
    phys_idx  = phys_df.set_index("inchi_key")

    # ── Physchem features ─────────────────────────────────────────────
    phys_cols = [c for c in phys_df.columns if c != "inchi_key"]

    def get_phys(iks):
        rows = []
        for ik in iks:
            if ik in phys_idx.index:
                rows.append(phys_idx.loc[ik, phys_cols].values)
            else:
                rows.append(np.full(len(phys_cols), np.nan))
        return np.array(rows, dtype=np.float32)

    pos_phys = get_phys(pos_iks.tolist())
    pmd_phys = pmd_df[phys_cols].values.astype(np.float32) if all(c in pmd_df for c in phys_cols) \
               else np.column_stack([
                   pmd_df.get(c, pd.Series(np.nan, index=pmd_df.index)).values
                   for c in phys_cols]).astype(np.float32)

    X_phys = np.vstack([pos_phys, pmd_phys])
    y_phys = np.array([1]*len(pos_phys) + [0]*len(pmd_phys))

    # Replace NaN with column means
    col_means = np.nanmean(X_phys, axis=0)
    for j in range(X_phys.shape[1]):
        mask = np.isnan(X_phys[:, j]); X_phys[mask, j] = col_means[j]

    lg.info(f"  Physchem matrix: {X_phys.shape} | pos={y_phys.sum()} neg={(y_phys==0).sum()}")

    # ── ECFP4 features ────────────────────────────────────────────────
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    def smiles_to_ecfp(smi: str) -> np.ndarray | None:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None: return None
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = mfpgen.GetFingerprint(mol)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    pos_ecfp_rows, pos_labels = [], []
    for ik in pos_iks:
        r = ik_to_row.get(ik, -1)
        if r >= 0:
            pos_ecfp_rows.append(ecfp_mat[r].astype(np.float32))
            pos_labels.append(1)

    pmd_ecfp_rows, neg_labels = [], []
    for _, row in pmd_df.iterrows():
        arr = smiles_to_ecfp(str(row.get("canonical_smiles", "")))
        if arr is not None:
            pmd_ecfp_rows.append(arr)
            neg_labels.append(0)

    n_min = min(len(pos_ecfp_rows), len(pmd_ecfp_rows))
    X_ecfp = np.vstack(pos_ecfp_rows[:n_min] + pmd_ecfp_rows[:n_min])
    y_ecfp = np.array([1]*n_min + [0]*n_min)
    lg.info(f"  ECFP4 matrix: {X_ecfp.shape}")

    # ── CV AUROC ──────────────────────────────────────────────────────
    lg.info("  Running 5-fold CV AUROC (physchem)...")
    phys_auroc, phys_std = cv_auroc(X_phys, y_phys)
    lg.info(f"  Physchem AUROC: {phys_auroc:.3f} ± {phys_std:.3f} "
            f"(limit ≤ {physchem_limit})")

    lg.info("  Running 5-fold CV AUROC (ECFP4)...")
    ecfp_auroc, ecfp_std = cv_auroc(X_ecfp, y_ecfp)
    lg.info(f"  ECFP4 AUROC:    {ecfp_auroc:.3f} ± {ecfp_std:.3f} "
            f"(limit ≤ {ecfp_limit})")

    phys_flag = phys_auroc > physchem_limit
    ecfp_flag = ecfp_auroc > ecfp_limit

    if phys_flag:
        lg.warning(f"  ⚠️  Physchem AUROC {phys_auroc:.3f} > {physchem_limit} limit. "
                   "PMD negatives may be too easy. Consider tightening property tolerances.")
    else:
        lg.info(f"  ✅ Physchem AUROC within limit")

    if ecfp_flag:
        lg.warning(f"  ⚠️  ECFP4 AUROC {ecfp_auroc:.3f} > {ecfp_limit} limit. "
                   "PMD negativs are too similar to positives in fingerprint space. "
                   "Consider lowering max_tanimoto_to_pos threshold.")
    else:
        lg.info(f"  ✅ ECFP4 AUROC within limit")

    # ── Check minimum counts ──────────────────────────────────────────
    min_required = pmd_cfg.get("min_pmd_negatives", 1000)
    count_flag = len(pmd_df) < min_required
    if count_flag:
        lg.warning(f"  ⚠️  PMD Count {len(pmd_df)} < {min_required} limit. ")
    else:
        lg.info(f"  ✅ PMD Count within limit ({len(pmd_df)} >= {min_required})")

    # ── Save results ──────────────────────────────────────────────────
    tables_dir = ROOT / cfg["outputs"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([
        {"probe": "physchem_only", "auroc_mean": round(phys_auroc,4),
         "auroc_std": round(phys_std,4), "limit": physchem_limit,
         "flag": phys_flag, "note": "AUXILIARY ONLY — not comparable to nnPU metrics"},
        {"probe": "ecfp4_only",    "auroc_mean": round(ecfp_auroc,4),
         "auroc_std": round(ecfp_std,4), "limit": ecfp_limit,
         "flag": ecfp_flag, "note": "AUXILIARY ONLY — not comparable to nnPU metrics"},
        {"probe": "pmd_count",     "auroc_mean": len(pmd_df),
         "auroc_std": 0.0,         "limit": min_required,
         "flag": count_flag, "note": "Minimum generated negative candidates"},
    ])
    results.to_csv(tables_dir / "phase4C_binary_eval.csv", index=False)
    lg.info(f"  Written: phase4C_binary_eval.csv")

    # Append to audit doc
    audit_path = ROOT / "docs" / "phase4C_negative_audit.md"
    audit_extra = f"""
## Binary Eval Artifact Audit Results

_Run: {ts}_

> **WARNING:** AUROC is used here for artifact auditing ONLY.
> These binary AUROC values MUST NOT appear as headline results.
> nnPU headline metrics are PU-Recall@k and Lift@k only.

| Probe | AUROC ± std | Limit | Status |
|---|---|---|---|
| Physchem-only | {phys_auroc:.3f} ± {phys_std:.3f} | ≤ {physchem_limit} | {"⚠️ FLAG" if phys_flag else "✅ PASS"} |
| ECFP4-only | {ecfp_auroc:.3f} ± {ecfp_std:.3f} | ≤ {ecfp_limit} | {"⚠️ FLAG" if ecfp_flag else "✅ PASS"} |
| Target Count | {len(pmd_df)} | ≥ {min_required} | {"⚠️ FLAG" if count_flag else "✅ PASS"} |
"""
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(audit_extra)
    lg.info("  Appended to docs/phase4C_negative_audit.md")

    lg.info("\n" + "=" * 70)
    lg.info("IFG-26 Phase 4C-ii Binary Eval — COMPLETE")
    lg.info(f"  Physchem AUROC: {phys_auroc:.3f} {'⚠️' if phys_flag else '✅'}")
    lg.info(f"  ECFP4 AUROC:    {ecfp_auroc:.3f} {'⚠️' if ecfp_flag else '✅'}")
    lg.info(f"  PMD Count:      {len(pmd_df)} {'⚠️' if count_flag else '✅'}")
    lg.info("=" * 70)

    # Write out the scientific validity status
    status_path = ROOT / "data" / "diagnostics" / "phase4C_scientific_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_data = {
        "physchem_pass": not bool(phys_flag),
        "ecfp4_pass": not bool(ecfp_flag),
        "count_pass": not bool(count_flag),
        "physchem_auroc": float(phys_auroc),
        "ecfp4_auroc": float(ecfp_auroc)
    }
    with open(status_path, "w") as f:
        json.dump(status_data, f, indent=2)

if __name__ == "__main__":
    main()
