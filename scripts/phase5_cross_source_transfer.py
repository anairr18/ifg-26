"""
phase5_cross_source_transfer.py
=================================
IFG-26 Phase 5 — Cross-Source Generalization Test.

Tests whether models trained on one database (MGDB or MGTbind) generalize
to the other. Measures performance collapse across:
    - Train MGDB → Test MGTbind
    - Train MGTbind → Test MGDB
    - Train Combined → Holdout (random 20%)

Uses ECFP4 features, Logistic Regression and Random Forest.

Outputs:
    data/phase5_cross_source_results.csv
    docs/phase5_cross_source.md

Usage:
    python scripts/phase5_cross_source_transfer.py [--config path] [--resume]
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase4_default.yaml"
SEED = 42


def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5_cross_source")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5_cross_source.log", encoding="utf-8"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg


def smiles_to_ecfp4(smiles_list):
    fps = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                fps.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)))
            else:
                fps.append([0] * 2048)
        except Exception:
            fps.append([0] * 2048)
    return np.array(fps, dtype=np.float32)


def recall_at_k(y_true, y_score, pct=0.05):
    k = max(1, int(len(y_score) * pct))
    top_idx = np.argsort(y_score)[::-1][:k]
    return float(np.sum(y_true[top_idx]) / max(1, np.sum(y_true)))


def eval_transfer(X_train, y_train, X_test, y_test, mdl_name, mdl, scale=False):
    if scale:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    mdl.fit(X_train, y_train)
    y_score = mdl.predict_proba(X_test)[:, 1]
    return {
        "model": mdl_name,
        "auroc": round(float(roc_auc_score(y_test, y_score)), 4),
        "auprc": round(float(average_precision_score(y_test, y_score)), 4),
        "recall_at_5pct": round(recall_at_k(y_test, y_score), 4),
    }


def load_source_smiles(csv_path: Path, smi_col: str, label: int, lg) -> pd.DataFrame:
    if not csv_path.exists():
        lg.warning(f"Not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path, low_memory=False)
    if smi_col not in df.columns:
        lg.warning(f"Column {smi_col} not in {csv_path}")
        return pd.DataFrame()
    df = df[[smi_col]].dropna().rename(columns={smi_col: "smiles"})
    df["label"] = label
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    out_path = ROOT / "data" / "phase5_cross_source_results.csv"

    if args.resume and out_path.exists():
        lg.info("phase5_cross_source_results.csv exists — skipping (--resume).")
        return

    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 5 — Cross-Source Transfer  {ts}")
    lg.info("=" * 70)

    curated = ROOT / "data" / "curated" / "phase1"
    pmd_neg = ROOT / "data" / "negatives" / "pmd_negatives.parquet"

    # Load positive ligands from each source
    mgdb_pos_df = load_source_smiles(
        curated / "ifg26_training_pairs.csv", "canonical_smiles", 1, lg
    )
    mgtbind_pos_df = load_source_smiles(
        curated / "mgtbind_compounds_canonicalized.csv", "canonical_smiles", 1, lg
    )

    # Load shared negatives
    if pmd_neg.exists():
        neg_df = pd.read_parquet(pmd_neg)
        neg_smi = neg_df["decoy_smiles"].dropna().tolist() if "decoy_smiles" in neg_df.columns else []
    else:
        neg_smi = []
        lg.warning("PMD negatives not found — using empty negative set. Results will be unreliable.")

    def make_dataset(pos_df: pd.DataFrame, neg_smiles: list) -> tuple:
        pos_smi = pos_df["smiles"].tolist() if not pos_df.empty else []
        n = min(len(pos_smi), len(neg_smiles))
        if n == 0:
            return None, None
        all_smi = pos_smi[:n] + neg_smiles[:n]
        y = np.array([1]*n + [0]*n)
        X = smiles_to_ecfp4(all_smi)
        return X, y

    models = {
        "LR": (LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"), True),
        "RF": (RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced"), False),
    }

    results = []

    X_mgdb, y_mgdb = make_dataset(mgdb_pos_df, neg_smi)
    X_mgtbind, y_mgtbind = make_dataset(mgtbind_pos_df, neg_smi)

    regimes = []
    if X_mgdb is not None and X_mgtbind is not None:
        regimes.append(("MGDB → MGTbind", X_mgdb, y_mgdb, X_mgtbind, y_mgtbind))
        regimes.append(("MGTbind → MGDB", X_mgtbind, y_mgtbind, X_mgdb, y_mgdb))

    # Combined → holdout
    if X_mgdb is not None and X_mgtbind is not None:
        X_combined = np.vstack([X_mgdb, X_mgtbind])
        y_combined = np.concatenate([y_mgdb, y_mgtbind])
        X_tr_comb, X_te_comb, y_tr_comb, y_te_comb = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=SEED, stratify=y_combined
        )
        regimes.append(("Combined → Holdout", X_tr_comb, y_tr_comb, X_te_comb, y_te_comb))

    for regime_name, X_tr, y_tr, X_te, y_te in regimes:
        lg.info(f"\n{regime_name}: train={len(y_tr)}, test={len(y_te)}")
        for mdl_name, (mdl, scale) in models.items():
            from sklearn.base import clone
            try:
                res = eval_transfer(X_tr, y_tr, X_te, y_te, mdl_name, clone(mdl), scale)
                res["regime"] = regime_name
                results.append(res)
                lg.info(f"  {mdl_name}: AUROC={res['auroc']}, Recall@5%={res['recall_at_5pct']}")
            except Exception as e:
                lg.warning(f"  {mdl_name} failed: {e}")

    df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    lg.info(f"\nWritten: {out_path.relative_to(ROOT)}")

    # ── Docs ──────────────────────────────────────────────────────────────
    table_rows = "\n".join([
        f"| {r['regime']} | {r['model']} | {r['auroc']} | {r.get('auprc', 'N/A')} | {r['recall_at_5pct']} |"
        for _, r in df.iterrows()
    ])

    doc_md = f"""# IFG-26 Phase 5 — Cross-Source Generalization

_Generated: {ts}_

## Results

| Training Regime | Model | AUROC | AUPRC | Recall@5% |
|---|---|---|---|---|
{table_rows}

## Interpretation

If AUROC in the **cross-source** settings (MGDB → MGTbind, MGTbind → MGDB) is significantly
lower than the combined-with-holdout setting, this indicates that the two databases populate
different chemical and biological spaces, and that models trained on one source may not
generalize to the other.

This is particularly relevant for molecular glue prediction: if model performance is
database-dependent, apparent benchmark scores may reflect the memorization of database-specific
biases rather than genuine molecular interaction principles.
"""
    with open(ROOT / "docs" / "phase5_cross_source.md", "w", encoding="utf-8") as f:
        f.write(doc_md)
    lg.info("Written: docs/phase5_cross_source.md")

    lg.info("=" * 70)
    lg.info("Phase 5 Cross-Source Transfer — COMPLETE")
    lg.info("=" * 70)


if __name__ == "__main__":
    main()
