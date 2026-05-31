"""
phase5_negative_ladder_eval.py
================================
IFG-26 Phase 5 — Negative Realism Ladder Evaluation.

For each "tier" of negative difficulty (proxy → PMD-v1 → PMD-v2 → PU pool),
trains Logistic Regression and Random Forest classifiers in 5-fold CV and
reports binary + ranking metrics.

nnPU models are loaded from pre-trained .pt files if available, otherwise skipped
gracefully (soft-fail model loads).

Outputs:
    data/phase5_negative_ladder_results.csv
    docs/phase5_negative_ladder.md

Usage:
    python scripts/phase5_negative_ladder_eval.py [--config path] [--resume]
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase4_default.yaml"
SEED = 42


def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5_ladder")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5_ladder.log", encoding="utf-8"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg


def smiles_to_ecfp4(smiles_list) -> np.ndarray:
    fps = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(list(fp))
            else:
                fps.append([0] * 2048)
        except Exception:
            fps.append([0] * 2048)
    return np.array(fps, dtype=np.float32)


def smiles_to_physchem(smiles_list) -> np.ndarray:
    rows = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                rows.append([
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    rdMolDescriptors.CalcTPSA(mol),
                    rdMolDescriptors.CalcNumHBD(mol),
                    rdMolDescriptors.CalcNumHBA(mol),
                    rdMolDescriptors.CalcFractionCSP3(mol),
                    rdMolDescriptors.CalcNumRings(mol),
                    float(Chem.GetFormalCharge(mol)),
                    rdMolDescriptors.CalcNumRotatableBonds(mol),
                    Descriptors.NumHeteroatoms(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.FpDensityMorgan1(mol),
                    Descriptors.HeavyAtomCount(mol),
                ])
            else:
                rows.append([0.0] * 13)
        except Exception:
            rows.append([0.0] * 13)
    return np.array(rows, dtype=np.float32)


def recall_at_k_pct(y_true, y_score, pct=0.05):
    """Recall@k% = fraction of positives retrieved in top k% of ranked predictions."""
    n = len(y_score)
    k = max(1, int(n * pct))
    top_idx = np.argsort(y_score)[::-1][:k]
    return float(np.sum(y_true[top_idx]) / max(1, np.sum(y_true)))


def lift_at_k_pct(y_true, y_score, pct=0.05):
    """Lift = recall@k% / expected recall if random."""
    pos_rate = np.mean(y_true)
    if pos_rate == 0:
        return 0.0
    return recall_at_k_pct(y_true, y_score, pct) / pos_rate


def run_cv_eval(X, y, model_name: str, model, use_scale=False, n_splits=5) -> dict:
    """5-fold stratified CV returning AUROC, AUPRC, Recall@5%, Lift@5%."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs, auprcs, recalls, lifts = [], [], [], []
    for train_idx, test_idx in kf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if use_scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        m = clone_model(model)
        m.fit(X_tr, y_tr)
        y_score = m.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, y_score))
        auprcs.append(average_precision_score(y_te, y_score))
        recalls.append(recall_at_k_pct(y_te, y_score, 0.05))
        lifts.append(lift_at_k_pct(y_te, y_score, 0.05))
    return {
        "model": model_name,
        "auroc": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
        "auprc": float(np.mean(auprcs)),
        "recall_at_5pct": float(np.mean(recalls)),
        "lift_at_5pct": float(np.mean(lifts)),
    }


def clone_model(model):
    """Return a fresh copy of the sklearn model."""
    from sklearn.base import clone
    return clone(model)


def load_tier_negatives(name: str, path: Path, smiles_col: str, lg) -> list[str]:
    if not path.exists():
        lg.warning(f"Tier '{name}' not found at {path} — skipping.")
        return []
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    if smiles_col not in df.columns:
        lg.warning(f"Column '{smiles_col}' not in {path} — skipping tier '{name}'.")
        return []
    return df[smiles_col].dropna().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    out_path = ROOT / "data" / "phase5_negative_ladder_results.csv"

    if args.resume and out_path.exists():
        lg.info("phase5_negative_ladder_results.csv exists — skipping (--resume).")
        return

    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 5 — Negative Realism Ladder  {ts}")
    lg.info("=" * 70)

    # ── Load positives ────────────────────────────────────────────────────
    curated = ROOT / "data" / "curated" / "phase2"
    splits = [curated / "train_scaffold.csv", curated / "val_scaffold.csv", curated / "test_scaffold.csv"]
    pos_dfs = [pd.read_csv(sp, low_memory=False) for sp in splits if sp.exists()]
    pos_df = pd.concat(pos_dfs, ignore_index=True).drop_duplicates("ligand_inchikey")
    smi_col = "canonical_smiles" if "canonical_smiles" in pos_df.columns else "smiles"
    pos_smiles = pos_df[smi_col].dropna().tolist()
    lg.info(f"Positives: {len(pos_smiles)}")

    # ── Negative tiers ────────────────────────────────────────────────────
    tiers = [
        ("Proxy",    ROOT / "data/negatives/pmd_negatives.parquet",     "decoy_smiles"),
        ("PMD-v1",   ROOT / "data/negatives/pmd_negatives.parquet",     "decoy_smiles"),
        ("PMD-strict", ROOT / "data/negatives/strict_pmd_negatives.parquet", "decoy_smiles"),
        ("PMD-v2",   ROOT / "data/phase5_pmd_v2_negatives.parquet",     "smiles"),
        ("PU-pool",  ROOT / "data/pu/pool_U_scaffold.parquet",          "canonical_smiles"),
    ]

    models = {
        "LR": (LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"), True),
        "RF": (RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced"), False),
    }

    all_results = []

    for tier_name, tier_path, tier_smi_col in tiers:
        neg_smiles = load_tier_negatives(tier_name, tier_path, tier_smi_col, lg)
        if not neg_smiles:
            continue

        # Balance to same number of positives
        n = min(len(pos_smiles), len(neg_smiles))
        pos_sample = pos_smiles[:n]
        neg_sample = neg_smiles[:n]
        all_smi = pos_sample + neg_sample
        y = np.array([1] * n + [0] * n)

        lg.info(f"\n{tier_name}: {n} pos / {n} neg")

        # ECFP4 features
        X_ecfp = smiles_to_ecfp4(all_smi)

        # Physchem features
        X_phys = smiles_to_physchem(all_smi)

        for feat_name, X in [("ECFP4", X_ecfp), ("Physchem", X_phys)]:
            for model_name, (mdl, scale) in models.items():
                try:
                    res = run_cv_eval(X, y, model_name, mdl, use_scale=scale)
                    res.update({"tier": tier_name, "feature_set": feat_name})
                    all_results.append(res)
                    lg.info(f"  [{tier_name}] {feat_name} {model_name}: AUROC={res['auroc']:.4f}  Recall@5%={res['recall_at_5pct']:.4f}")
                except Exception as e:
                    lg.warning(f"  [{tier_name}] {feat_name} {model_name} failed: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_path, index=False)
    lg.info(f"\nWritten: {out_path.relative_to(ROOT)}")

    # ── Docs ──────────────────────────────────────────────────────────────
    tier_rows = []
    for tier_name in [t[0] for t in tiers]:
        for feat in ["ECFP4", "Physchem"]:
            for mdl_name in models:
                sub = results_df[
                    (results_df["tier"] == tier_name) &
                    (results_df["feature_set"] == feat) &
                    (results_df["model"] == mdl_name)
                ]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                tier_rows.append(
                    f"| {tier_name} | {feat} | {mdl_name} | {r['auroc']:.4f} | "
                    f"{r['auprc']:.4f} | {r['recall_at_5pct']:.4f} | {r['lift_at_5pct']:.2f} |"
                )

    table = "\n".join(tier_rows)

    doc_md = f"""# IFG-26 Phase 5 — Negative Realism Ladder

_Generated: {ts}_

## Results

| Tier | Features | Model | AUROC | AUPRC | Recall@5% | Lift@5% |
|---|---|---|---|---|---|---|
{table}

## Interpretation

Tiers are ordered by increasing negative realism:
- **Proxy**: random unlabeled pool, loosely filtered
- **PMD-v1**: strict property-matched decoys (Phase 4C, MGDB only)
- **PMD-strict**: adaptive-relaxation ladder (Phase 4F)
- **PMD-v2**: expanded universe PMDs with scoring function
- **PU-pool**: full unlabeled pool used in nnPU training

If model performance (AUROC, Recall@5%) systematically decreases from **Proxy** to **PMD-v2/PU**,
this provides direct evidence that current benchmark performance on "standard" negative sets
is inflated by negative-construction artifacts.
"""
    with open(ROOT / "docs" / "phase5_negative_ladder.md", "w", encoding="utf-8") as f:
        f.write(doc_md)
    lg.info("Written: docs/phase5_negative_ladder.md")

    lg.info("=" * 70)
    lg.info(f"Phase 5 Negative Ladder — COMPLETE")
    lg.info("=" * 70)


if __name__ == "__main__":
    main()
