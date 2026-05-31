"""
phase3_baseline_probes.py
=========================
IFG-26 Phase 3B — Baseline Shortcut Probes.

Trains three lightweight probes on positives vs proxy-negatives,
saves score distributions by similarity bin, updates docs/phase3_shortcut_audit.md.

Config:   configs/experiment/phase3_default.yaml
Usage:    python scripts/phase3_baseline_probes.py [--config <path>]

Probes:
    L0 — Ligand-only:  ECFP4 → LogisticRegression
    P0 — Protein-only: label IDs one-hot → LogisticRegression
    LP0 — Ligand+Protein: ECFP4 + protein labels → LogisticRegression

Proxy negative strategy:
    MGDB canonicalized compounds NOT present in training pairs (MGDBunmapped).
    Labeled as negative=0. Positives are training pairs labeled 1.
    Cap at max_proxy_negatives for class balance.

NOTE: All probes are proxy evaluations for Phase 3. PU-calibrated evaluation
      deferred to Phase 4. Probe results are explicitly labeled as proxy.
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              precision_recall_curve, roc_curve)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase3_default.yaml"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase3_probes")
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(log_dir / "phase3_probes.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    if not lg.handlers:
        lg.addHandler(fh)
        lg.addHandler(sh)
    return lg


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Load features
# ---------------------------------------------------------------------------
def load_features(features_dir: Path, logger: logging.Logger):
    logger.info("Loading feature matrices from cache...")
    lig_idx  = pd.read_parquet(features_dir / "ligand_index.parquet")
    ecfp_mat = np.load(str(features_dir / "ligands_ecfp4.npy"))
    phys_df  = pd.read_parquet(features_dir / "ligands_physchem.parquet")
    p_idx    = pd.read_parquet(features_dir / "protein_index.parquet")
    pairs    = pd.read_parquet(features_dir / "scaffold_pairs_index.parquet")
    bins_df  = pd.read_parquet(features_dir / "test_similarity_bins.parquet")
    logger.info(f"  Ligands: {len(lig_idx)} | ECFP4: {ecfp_mat.shape} | "
                f"Proteins: {len(p_idx)} | Pairs: {len(pairs)}")
    return lig_idx, ecfp_mat, phys_df, p_idx, pairs, bins_df


# ---------------------------------------------------------------------------
# Proxy negatives
# ---------------------------------------------------------------------------
def build_proxy_negatives(pairs_df: pd.DataFrame,
                          mgdb_canon_path: Path,
                          max_neg: int, seed: int,
                          ecfp_radius: int, ecfp_nbits: int,
                          logger: logging.Logger):
    """
    Proxy negatives = MGDB canonicalized compounds with valid ECFP4 but NOT
    in any training/val/test positive pair (by InChIKey). Computes ECFP4 from
    SMILES directly — does NOT require the feature cache to contain them.
    """
    pos_iks = set(pairs_df["ligand_inchikey"].dropna().unique())
    mgdb = pd.read_csv(mgdb_canon_path, encoding="utf-8", low_memory=False)
    mgdb = mgdb[mgdb["canonicalized_ok"] == True].copy()
    mgdb_cands = mgdb[~mgdb["inchi_key"].isin(pos_iks)].dropna(subset=["inchi_key"])
    logger.info(f"  MGDB compounds not in training pairs: {len(mgdb_cands)}")

    rng = np.random.default_rng(seed)
    if len(mgdb_cands) > max_neg * 2:  # oversample then filter
        mgdb_cands = mgdb_cands.sample(max_neg * 2, random_state=seed)

    from rdkit.Chem import rdFingerprintGenerator
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=ecfp_radius, fpSize=ecfp_nbits)
    neg_fps, neg_iks = [], []
    for _, row in mgdb_cands.iterrows():
        smi = str(row.get("canonical_smiles", ""))
        if not smi or smi.lower() in ("nan", ""):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = mfpgen.GetFingerprint(mol)
        arr = np.zeros(ecfp_nbits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        neg_fps.append(arr)
        neg_iks.append(row["inchi_key"])
        if len(neg_fps) >= max_neg:
            break

    if not neg_fps:
        # Absolute fallback: generate zeros (edge case only)
        logger.warning("  No proxy negatives found! Using zero-vectors as fallback.")
        neg_mat = np.zeros((1, ecfp_nbits), dtype=np.float32)
        return neg_mat, []

    neg_mat = np.array(neg_fps, dtype=np.float32)
    logger.info(f"  Using {len(neg_mat)} proxy negatives")
    return neg_mat, neg_iks


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
def get_pair_ecfp(pairs_df: pd.DataFrame, ecfp_mat: np.ndarray,
                  split: str) -> np.ndarray:
    """ECFP4 for all pairs in a split. Rows missing in feature cache → zero vector."""
    sub = pairs_df[pairs_df["split"] == split]
    X = ecfp_mat[sub["ligand_feature_row"].clip(0).values].astype(np.float32)
    return X


def get_protein_onehot(pairs_df: pd.DataFrame, p_idx: pd.DataFrame,
                       split: str) -> np.ndarray:
    """Concatenated [e3_label_id, target_label_id] one-hot encoded."""
    sub = pairs_df[pairs_df["split"] == split]
    n_proteins = len(p_idx)
    # e3 one-hot
    e3_oh  = np.eye(n_proteins + 1, dtype=np.float32)[
        sub["e3_label_id"].clip(-1).values + 1]   # shift -1→0
    tgt_oh = np.eye(n_proteins + 1, dtype=np.float32)[
        sub["target_label_id"].clip(-1).values + 1]
    return np.hstack([e3_oh, tgt_oh])


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------
def run_probe(
    probe_name: str,
    X_train, y_train,
    X_val,   pairs_val,   bins_df: pd.DataFrame,
    X_test,  pairs_test,
    cfg: dict, logger: logging.Logger,
    probe_dir: Path,
    features_dir: Path,
):
    """Train one probe and save model + scores + bin-stratified report."""
    logger.info(f"\n  [{probe_name}] Training on {X_train.shape[0]} samples "
                f"(pos={y_train.sum()}, neg={len(y_train)-y_train.sum()})...")

    lr_cfg = cfg.get("probes", {}).get("logistic", {})
    model = LogisticRegression(
        C=lr_cfg.get("C", 1.0),
        max_iter=lr_cfg.get("max_iter", 1000),
        solver=lr_cfg.get("solver", "lbfgs"),
        n_jobs=-1,
        random_state=cfg.get("probes", {}).get("seed", 42),
    )
    model.fit(X_train, y_train)

    # Score distributions (probability of class 1)
    def scores(X):
        return model.predict_proba(X)[:, 1]

    train_scores = scores(X_train)
    val_scores   = scores(X_val)
    test_scores  = scores(X_test)

    # Metrics (proxy — not PU-calibrated)
    def metrics(y, s, label):
        if len(np.unique(y)) < 2:
            return {"auroc": None, "auprc": None, "note": "single_class"}
        return {
            "auroc":  round(float(roc_auc_score(y, s)), 4),
            "auprc":  round(float(average_precision_score(y, s)), 4),
            "n":      int(len(y)),
            "n_pos":  int(y.sum()),
            "label":  label,
        }

    # For val/test, only positives are scored (proxy eval)
    val_m  = {"label": "val",  "n_positives": len(pairs_val),
               "score_mean": round(float(val_scores.mean()), 4),
               "score_std":  round(float(val_scores.std()),  4),
               "score_median": round(float(np.median(val_scores)), 4)}
    test_m = {"label": "test", "n_positives": len(pairs_test),
               "score_mean": round(float(test_scores.mean()), 4),
               "score_std":  round(float(test_scores.std()),  4),
               "score_median": round(float(np.median(test_scores)), 4)}

    # Similarity-bin-stratified scores for test set
    bin_map = bins_df.set_index("compound_inchi_key")["similarity_bin"].to_dict()
    test_iks = pairs_test["ligand_inchikey"].values
    test_bins = np.array([bin_map.get(ik, "?") for ik in test_iks])
    bin_stats = {}
    for b in ["A", "B", "C", "D", "E"]:
        mask = test_bins == b
        if mask.sum() > 0:
            sub_scores = test_scores[mask]
            bin_stats[b] = {
                "n": int(mask.sum()),
                "median": round(float(np.median(sub_scores)), 4),
                "q25":    round(float(np.percentile(sub_scores, 25)), 4),
                "q75":    round(float(np.percentile(sub_scores, 75)), 4),
                "mean":   round(float(sub_scores.mean()), 4),
            }
        else:
            bin_stats[b] = {"n": 0}

    logger.info(f"  [{probe_name}] Val median score: {val_m['score_median']:.3f} | "
                f"Test median: {test_m['score_median']:.3f}")
    logger.info(f"  [{probe_name}] Bin-stratified test scores: "
                + " | ".join(f"{b}: med={bin_stats[b].get('median','—')}" for b in "ABCDE"))

    # Save model
    probe_dir.mkdir(parents=True, exist_ok=True)
    with open(probe_dir / f"{probe_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scores for all three splits
    results = {
        "probe_name":   probe_name,
        "val_metrics":  val_m,
        "test_metrics": test_m,
        "test_bin_stratified": bin_stats,
        "editorial_policy_2": (
            "All headline results reported both overall and excluding Bin E (NN>0.95) "
            "as a sensitivity analysis per IFG-26 Phase 3 editorial policy."
        ),
    }

    # Exclude-bin-E summary
    e_mask     = test_bins == "E"
    notE_mask  = ~e_mask
    if notE_mask.sum() > 0:
        results["test_exclude_binE"] = {
            "n": int(notE_mask.sum()),
            "score_mean":   round(float(test_scores[notE_mask].mean()), 4),
            "score_median": round(float(np.median(test_scores[notE_mask])), 4),
        }
    results["test_binE_only"] = {
        "n": int(e_mask.sum()),
        "score_mean": round(float(test_scores[e_mask].mean()), 4) if e_mask.sum() else None,
    }

    with open(features_dir / f"probe_{probe_name}_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)

    # Score distribution plot (val + test)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(val_scores,  bins=40, alpha=0.65, color="#1976D2", label=f"Val (n={len(val_scores)})")
    ax.hist(test_scores, bins=40, alpha=0.65, color="#E64A19", label=f"Test (n={len(test_scores)})")
    ax.set_xlabel(f"Probe {probe_name} Score (P(positive))", fontsize=11)
    ax.set_ylabel("Pairs", fontsize=11)
    ax.set_title(f"Probe {probe_name} — Score Distribution (Positives Only)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_dir = ROOT / "results" / "figures" / "phase3"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"probe_{probe_name}_scores.png", dpi=150)
    plt.close()
    logger.info(f"  [{probe_name}] Saved scores + model + plot.")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg_path = Path(args.config)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger = setup_logging(ROOT / "logs")
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 70)
    logger.info(f"IFG-26 Phase 3B — Baseline Probes  {ts}")
    logger.info("=" * 70)

    features_dir = ROOT / cfg["outputs"]["features_dir"]
    probe_dir    = features_dir / "probes"

    lig_idx, ecfp_mat, phys_df, p_idx, pairs, bins_df = load_features(features_dir, logger)

    # ── Proxy negatives ───────────────────────────────────────────────
    logger.info("\n--- Proxy Negatives ---")
    probe_cfg = cfg.get("probes", {})
    seed      = probe_cfg.get("seed", 42)
    max_neg   = probe_cfg.get("max_proxy_negatives", 5000)
    ecfp_cfg = cfg.get("ligand_features", {}).get("ecfp4", {})
    neg_mat, neg_iks = build_proxy_negatives(
        pairs,
        ROOT / cfg["inputs"]["mgdb_canon"],
        max_neg, seed,
        ecfp_radius=ecfp_cfg.get("radius", 2),
        ecfp_nbits=ecfp_cfg.get("nbits", 2048),
        logger=logger,
    )

    # ── Split pairs by partition ──────────────────────────────────────
    train_pos = pairs[pairs["split"] == "train"]
    val_pos   = pairs[pairs["split"] == "val"]
    test_pos  = pairs[pairs["split"] == "test"]

    # ECFP4 for positives
    X_train_pos_ecfp = ecfp_mat[train_pos["ligand_feature_row"].clip(0).values].astype(np.float32)
    X_val_ecfp       = ecfp_mat[val_pos["ligand_feature_row"].clip(0).values].astype(np.float32)
    X_test_ecfp      = ecfp_mat[test_pos["ligand_feature_row"].clip(0).values].astype(np.float32)

    # Build balanced training set: positives + proxy negatives
    n_pos   = len(X_train_pos_ecfp)
    n_neg   = len(neg_mat)
    X_train = np.vstack([X_train_pos_ecfp, neg_mat]).astype(np.float32)
    y_train = np.array([1] * n_pos + [0] * n_neg, dtype=np.int32)
    logger.info(f"  Train set: {n_pos} pos + {n_neg} neg = {len(y_train)}")

    # ── Probe L0: Ligand-only (ECFP4) ────────────────────────────────
    logger.info("\n--- Probe L0: Ligand-only (ECFP4) ---")
    results_l0 = run_probe(
        "L0", X_train, y_train,
        X_val_ecfp,  val_pos,  bins_df,
        X_test_ecfp, test_pos,
        cfg, logger, probe_dir, features_dir,
    )

    # ── Probe P0: Protein-only (one-hot E3+Target) ────────────────────
    logger.info("\n--- Probe P0: Protein-only (one-hot label IDs) ---")
    n_proteins = len(p_idx)

    def build_protein_features(df_pos, neg_iks_count=None):
        n_p1 = n_proteins + 1
        e3_oh  = np.eye(n_p1, dtype=np.float32)[
            df_pos["e3_label_id"].clip(-1).values + 1]
        tgt_oh = np.eye(n_p1, dtype=np.float32)[
            df_pos["target_label_id"].clip(-1).values + 1]
        return np.hstack([e3_oh, tgt_oh])

    X_train_prot_pos = build_protein_features(train_pos)
    # Proxy negatives have unknown proteins → use zero vectors
    X_train_prot_neg = np.zeros((n_neg, 2 * (n_proteins + 1)), dtype=np.float32)
    X_train_prot     = np.vstack([X_train_prot_pos, X_train_prot_neg])

    X_val_prot  = build_protein_features(val_pos)
    X_test_prot = build_protein_features(test_pos)

    results_p0 = run_probe(
        "P0", X_train_prot, y_train,
        X_val_prot,  val_pos,  bins_df,
        X_test_prot, test_pos,
        cfg, logger, probe_dir, features_dir,
    )

    # ── Probe LP0: Ligand + Protein concat ────────────────────────────
    logger.info("\n--- Probe LP0: Ligand + Protein ---")
    X_train_lp = np.hstack([X_train, X_train_prot])
    X_val_lp   = np.hstack([X_val_ecfp, X_val_prot])
    X_test_lp  = np.hstack([X_test_ecfp, X_test_prot])

    results_lp0 = run_probe(
        "LP0", X_train_lp, y_train,
        X_val_lp,  val_pos,  bins_df,
        X_test_lp, test_pos,
        cfg, logger, probe_dir, features_dir,
    )

    # ── Shortcut audit doc ───────────────────────────────────────────
    logger.info("\n--- Writing phase3_shortcut_audit.md ---")
    ep = cfg.get("editorial_policies", {})
    p1 = ep.get("policy_1", "")
    p2 = ep.get("policy_2", "")

    # Load featurize summary
    feat_summary_path = features_dir / "phase3_featurize_summary.json"
    feat_sum = json.loads(feat_summary_path.read_text()) if feat_summary_path.exists() else {}
    bins = feat_sum.get("test_similarity_bins", {})

    def bin_row(b, probe_name, results):
        bs = results.get("test_bin_stratified", {}).get(b, {})
        n   = bs.get("n", 0)
        med = bs.get("median", "—")
        q25 = bs.get("q25", "—")
        q75 = bs.get("q75", "—")
        return f"| {b} | {n} | {med} | {q25}–{q75} |"

    def probe_overview(name, r):
        return (f"| {name} | {r.get('val_metrics',{}).get('score_median','—')} | "
                f"{r.get('test_metrics',{}).get('score_median','—')} | "
                f"{r.get('test_exclude_binE',{}).get('score_median','—')} |")

    shortcut_md = f"""# IFG-26 Phase 3 — Shortcut Audit & Editorial Policy

_Generated: {ts}_

---

## Editorial Policies (Locked, Phase 3+)

> **Policy 1:** {p1}

> **Policy 2:** {p2}

These two policies are embedded as structured fields in
`configs/experiment/phase3_default.yaml` and will propagate to all downstream
Phase 4 training and evaluation reports automatically.

---

## Feature Cache Summary

| Item | Value |
|---|---|
| Unique ligands (InChIKey) | {feat_sum.get('n_unique_ligands', '?')} |
| ECFP4 matrix shape | {feat_sum.get('ecfp4_shape', '?')} |
| Physchem descriptors | {len(feat_sum.get('physchem_cols', []))} |
| Unique proteins (UniProt) | {feat_sum.get('n_unique_proteins', '?')} |
| Protein embedding method | Tier P1: label encoding (Phase 3); P2 ESM2 deferred to Phase 4 |
| Scaffold train/val/test | {feat_sum.get('n_train_rows','?')} / {feat_sum.get('n_val_rows','?')} / {feat_sum.get('n_test_rows','?')} |

---

## Test-Set Similarity Bins

ECFP4 NN Tanimoto of each test ligand against all train ligands.

| Bin | NN Range | Test Ligands | % of Test |
|---|---|---|---|
| A | [0.00, 0.60) | {bins.get('A', 0)} | {100*bins.get('A',0)/max(sum(bins.values()),1):.1f}% |
| B | [0.60, 0.80) | {bins.get('B', 0)} | {100*bins.get('B',0)/max(sum(bins.values()),1):.1f}% |
| C | [0.80, 0.85) | {bins.get('C', 0)} | {100*bins.get('C',0)/max(sum(bins.values()),1):.1f}% |
| D | [0.85, 0.95) | {bins.get('D', 0)} | {100*bins.get('D',0)/max(sum(bins.values()),1):.1f}% |
| **E** | **[0.95, 1.00]** | **{bins.get('E', 0)}** | **{100*bins.get('E',0)/max(sum(bins.values()),1):.1f}%** |

Bin E compounds (NN ≥ 0.95) are analog series of training positives. Per
Policy 2, all headline numbers must be reported both overall and with Bin E
excluded as a sensitivity analysis.

---

## Probe Overview (Proxy Evaluation — PU calibration in Phase 4)

> **NOTE:** Scores are probabilities output by LogisticRegression trained on
> positives vs MGDB proxy negatives. These are NOT PU-calibrated AUROC/AUPRC.
> They quantify shortcut risk: if P0 (protein-only) scores nearly as high as
> L0 (ligand-only), the model is exploiting protein identity shortcuts.

| Probe | Val Median Score | Test Median Score | Test excl. Bin E |
|---|---|---|---|
{probe_overview('L0 (ECFP4)', results_l0)}
{probe_overview('P0 (protein)', results_p0)}
{probe_overview('LP0 (ECFP4+protein)', results_lp0)}

### Similarity-Stratified Test Scores

#### L0 (Ligand-only, ECFP4)
| Bin | N test | Median | IQR |
|---|---|---|---|
{chr(10).join(bin_row(b, 'L0', results_l0) for b in 'ABCDE')}

#### P0 (Protein-only)
| Bin | N test | Median | IQR |
|---|---|---|---|
{chr(10).join(bin_row(b, 'P0', results_p0) for b in 'ABCDE')}

#### LP0 (Ligand + Protein)
| Bin | N test | Median | IQR |
|---|---|---|---|
{chr(10).join(bin_row(b, 'LP0', results_lp0) for b in 'ABCDE')}

---

## Red Flags to Monitor

Reviewers will look for these patterns. Check Phase 4 results against them:

1. **P0 ≈ L0** — protein-only probe matches ligand-only → protein identity shortcut
2. **Score explodes in Bin E only** — model is exploiting near-duplicate training analogs
3. **Val >> Test gap** — scaffold split val is harder than expected; train overfit
4. **Mutation rows score lower** — mutant targets under-represented (check Phase 4 strat.)

---

## Plots

- `results/figures/phase3/test_nn_tanimoto_hist.png`
- `results/figures/phase3/test_nn_bin_counts.png`
- `results/figures/phase3/probe_L0_scores.png`
- `results/figures/phase3/probe_P0_scores.png`
- `results/figures/phase3/probe_LP0_scores.png`
"""

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    with open(docs_dir / "phase3_shortcut_audit.md", "w", encoding="utf-8") as f:
        f.write(shortcut_md)
    logger.info("  Written: docs/phase3_shortcut_audit.md")

    # ── Final ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("IFG-26 Phase 3B Baseline Probes — COMPLETE")
    logger.info(f"  L0 test median: {results_l0['test_metrics']['score_median']}")
    logger.info(f"  P0 test median: {results_p0['test_metrics']['score_median']}")
    logger.info(f"  LP0 test median: {results_lp0['test_metrics']['score_median']}")
    logger.info("=" * 70)
    logger.info("\nNext step: proceed to Phase 4 training (phase4_train.py)")


if __name__ == "__main__":
    main()
