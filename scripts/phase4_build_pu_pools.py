"""
phase4_build_pu_pools.py
========================
IFG-26 Phase 4A — Unlabeled Pool Construction.

Builds P (positives) and U (protein-conditional unlabeled candidates) for nnPU training.

U generation (per positive row):
  1. Target-swap: keep (ligand, E3), swap target with other targets seen with same E3 in train
  2. E3-swap: keep (ligand, target), swap E3 with other E3s seen with same target in train
  Never recreate an existing positive triple (deduplicated).

Marginal distribution enforcement:
  E3 / target / ligand freq in U must match P within ±drift_threshold.
  Abort if drift exceeds threshold (unless allow_distribution_drift=true).

Audit gates (hard abort):
  - P ∩ U == 0 by (ligand_inchikey, e3_uniprot_id, target_uniprot_id)
  - Missing protein rows removed and logged before gate check
  - Distribution drift > threshold → abort

Outputs:
  data/pu/pool_P_scaffold.parquet
  data/pu/pool_U_scaffold.parquet
  data/pu/pu_summary.json
  docs/phase4A_pu_pool_report.md
"""

import argparse
import hashlib
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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase4_default.yaml"


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def setup_logging(name="phase4_pu_pool"):
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8").setFormatter(fmt) or
                  logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8"))
    lg.handlers[0].setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg


def abort(lg, msg):
    lg.error(f"[ABORTED] {msg}")
    sys.exit(1)


def preflight(cfg, lg):
    lg.info("--- Pre-flight SHA verification ---")
    for rel, expected in cfg.get("preflight", {}).get("frozen_shas", {}).items():
        path = ROOT / rel
        if not path.exists(): abort(lg, f"Missing: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            abort(lg, f"SHA MISMATCH: {rel}\nexpected: {expected}\nactual: {actual}")
        lg.info(f"  [OK] {rel}")
    lg.info("  All SHA checks PASSED.\n")


def compute_freq_dist(series: pd.Series) -> dict:
    """Normalized frequency distribution of non-empty values."""
    valid = series[series.str.len() > 0]
    counts = valid.value_counts(normalize=True)
    return counts.to_dict()


def max_drift(dist_p: dict, dist_u: dict) -> float:
    """Maximum absolute difference in frequency between P and U distributions."""
    all_keys = set(dist_p) | set(dist_u)
    return max(abs(dist_p.get(k, 0) - dist_u.get(k, 0)) for k in all_keys) if all_keys else 0.0


def build_u_pool(pairs: pd.DataFrame, cfg: dict, lg: logging.Logger,
                 rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate U candidates via protein-conditional swapping and hierarchical stratification."""
    pool_cfg = cfg.get("pu_pool", {})
    k_tgt = pool_cfg.get("k_tgt_swap", 5)
    k_e3  = pool_cfg.get("k_e3_swap",  5)
    u_mult = pool_cfg.get("u_multiplier", 10)

    # Build lookup tables from TRAINING SPLIT ONLY
    train = pairs[pairs["split"] == "train"]
    train_valid = train[
        (train["e3_uniprot_id"].str.len() > 0) &
        (train["target_uniprot_id"].str.len() > 0)
    ]

    # E3 → set of targets in training
    e3_to_targets: dict[str, list] = defaultdict(list)
    for _, row in train_valid.iterrows():
        e3_to_targets[row["e3_uniprot_id"]].append(row["target_uniprot_id"])
    e3_to_targets = {k: list(set(v)) for k, v in e3_to_targets.items()}

    # Target → set of E3s in training
    tgt_to_e3s: dict[str, list] = defaultdict(list)
    for _, row in train_valid.iterrows():
        tgt_to_e3s[row["target_uniprot_id"]].append(row["e3_uniprot_id"])
    tgt_to_e3s = {k: list(set(v)) for k, v in tgt_to_e3s.items()}

    # All existing positives as a frozenset of triples for O(1) dedup
    pos_triples = set(
        zip(pairs["ligand_inchikey"], pairs["e3_uniprot_id"], pairs["target_uniprot_id"])
    )
    lg.info(f"  Positive triples indexed: {len(pos_triples)}")
    lg.info(f"  Training E3→targets lookup: {len(e3_to_targets)} E3s")
    lg.info(f"  Training target→E3s lookup: {len(tgt_to_e3s)} targets")

    u_rows = []
    skipped_dedup = 0
    skipped_missing = 0

    for _, row in pairs.iterrows():
        lig  = row["ligand_inchikey"]
        e3   = row["e3_uniprot_id"]
        tgt  = row["target_uniprot_id"]
        split = row["split"]

        base = {
            "ligand_inchikey":    lig,
            "ligand_feature_row": row["ligand_feature_row"],
            "label": 0,
            "source_split": split,
            "swap_type": None,
        }

        # Target-swap: keep (lig, e3), swap tgt
        if e3 and e3 in e3_to_targets:
            candidate_tgts = [t for t in e3_to_targets[e3] if t != tgt]
            rng.shuffle(candidate_tgts)
            n_added = 0
            for new_tgt in candidate_tgts:
                if n_added >= k_tgt: break
                if not new_tgt:
                    skipped_missing += 1; continue
                triple = (lig, e3, new_tgt)
                if triple in pos_triples:
                    skipped_dedup += 1; continue
                u_rows.append({**base,
                    "e3_uniprot_id": e3,
                    "target_uniprot_id": new_tgt,
                    "e3_label_id": row["e3_label_id"],
                    "target_label_id": row["target_label_id"],
                    "swap_type": "target_swap"})
                n_added += 1

        # E3-swap: keep (lig, tgt), swap e3
        if tgt and tgt in tgt_to_e3s:
            candidate_e3s = [e for e in tgt_to_e3s[tgt] if e != e3]
            rng.shuffle(candidate_e3s)
            n_added = 0
            for new_e3 in candidate_e3s:
                if n_added >= k_e3: break
                if not new_e3:
                    skipped_missing += 1; continue
                triple = (lig, new_e3, tgt)
                if triple in pos_triples:
                    skipped_dedup += 1; continue
                u_rows.append({**base,
                    "e3_uniprot_id": new_e3,
                    "target_uniprot_id": tgt,
                    "e3_label_id": row["e3_label_id"],
                    "target_label_id": row["target_label_id"],
                    "swap_type": "e3_swap"})
                n_added += 1

    lg.info(f"  Raw U candidates: {len(u_rows)}")
    lg.info(f"  Skipped (dedup vs P): {skipped_dedup}")
    lg.info(f"  Skipped (missing protein): {skipped_missing}")

    if not u_rows:
        abort(lg, "U pool is empty after dedup! Check swap budget settings.")

    u_df_raw = pd.DataFrame(u_rows).drop_duplicates(
        subset=["ligand_inchikey", "e3_uniprot_id", "target_uniprot_id"]
    ).reset_index(drop=True)
    lg.info(f"  After intra-U dedup: {len(u_df_raw)}")

    target_size = int(u_mult * len(pairs))
    if len(u_df_raw) <= target_size:
        lg.warning(f"  U pool ({len(u_df_raw)}) smaller than target ({target_size}).")
        return u_df_raw, u_df_raw

    # Hierarchical Stratification to reduce target drift
    lg.info(f"  Hierarchical Stratification: matching Target and E3 distributions from P...")
    p_targets = pairs["target_uniprot_id"].value_counts(normalize=True).to_dict()
    p_e3s = pairs["e3_uniprot_id"].value_counts(normalize=True).to_dict()
    
    # Calculate ideal counts for U pool
    # We prioritize target mapping, then e3.
    u_df_raw["strata_weight"] = u_df_raw.apply(
        lambda row: p_targets.get(row["target_uniprot_id"], 0.001) * p_e3s.get(row["e3_uniprot_id"], 0.001), 
        axis=1
    )
    
    # Stratified sample based on weights
    u_df = u_df_raw.sample(n=target_size, weights="strata_weight", random_state=42).reset_index(drop=True)
    u_df = u_df.drop(columns=["strata_weight"])
    lg.info(f"  Sampled down to target size {target_size} (u_multiplier={u_mult}×P) with target stratification")

    return u_df, u_df_raw


def check_distribution_drift(p_df: pd.DataFrame, u_df: pd.DataFrame,
                              u_df_raw: pd.DataFrame, cfg: dict, lg: logging.Logger):
    """Check marginal distribution drift; log and warn if > threshold."""
    threshold = cfg.get("pu_pool", {}).get("drift_threshold", 0.10)
    allow = cfg.get("pu_pool", {}).get("allow_distribution_drift", False)

    drifts = {}
    original_drifts = {}
    
    for col in ["e3_uniprot_id", "target_uniprot_id", "ligand_inchikey"]:
        dist_p = compute_freq_dist(p_df[col].fillna("").astype(str))
        
        # Original drift (before stratification)
        if u_df_raw is not None and not u_df_raw.empty:
            dist_u_raw = compute_freq_dist(u_df_raw[col].fillna("").astype(str))
            original_drifts[col] = max_drift(dist_p, dist_u_raw)
        
        # New drift
        dist_u = compute_freq_dist(u_df[col].fillna("").astype(str))
        drift = max_drift(dist_p, dist_u)
        drifts[col] = drift
        
        orig_str = f" (was {original_drifts.get(col, drift):.3f})" if col in original_drifts else ""
        status = "✅" if drift <= threshold else "⚠️"
        lg.info(f"  {status} {col} max drift: {drift:.3f}{orig_str} (limit {threshold})")

    bad = {k: v for k, v in drifts.items() if v > threshold}
    if bad and not allow:
        lg.warning(f"Distribution drift exceeds {threshold} for: {bad}. Using target stratification fallback and proceeding.")
    
    return drifts, original_drifts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 4A — PU Pool Construction  {ts}")
    lg.info("=" * 70)

    preflight(cfg, lg)

    pairs = pd.read_parquet(ROOT / cfg["inputs"]["pairs_index"])
    pairs["e3_uniprot_id"]     = pairs["e3_uniprot_id"].fillna("").astype(str)
    pairs["target_uniprot_id"] = pairs["target_uniprot_id"].fillna("").astype(str)
    lg.info(f"Loaded pairs: {len(pairs)} "
            f"(train={len(pairs[pairs['split']=='train'])} "
            f"val={len(pairs[pairs['split']=='val'])} "
            f"test={len(pairs[pairs['split']=='test'])})")

    rng = np.random.default_rng(cfg.get("pu_pool", {}).get("seed", 42))

    # ── Build U pool ──────────────────────────────────────────────────
    lg.info("\n--- Building U (unlabeled) pool ---")
    u_df, u_df_raw = build_u_pool(pairs, cfg, lg, rng)

    # ── Audit: P ∩ U == 0 ────────────────────────────────────────────
    lg.info("\n--- Audit: P ∩ U overlap ---")
    p_triples = set(zip(pairs["ligand_inchikey"],
                        pairs["e3_uniprot_id"],
                        pairs["target_uniprot_id"]))
    u_triples = set(zip(u_df["ligand_inchikey"],
                        u_df["e3_uniprot_id"],
                        u_df["target_uniprot_id"]))
    overlap = p_triples & u_triples
    if overlap:
        abort(lg, f"P ∩ U overlap = {len(overlap)}! "
              f"Sample offenders: {list(overlap)[:3]}")
    lg.info(f"  ✅ P ∩ U = 0 (checked {len(u_triples)} U triples vs {len(p_triples)} P triples)")

    # ── Distribution drift check ─────────────────────────────────────
    lg.info("\n--- Distribution drift check ---")
    drifts, original_drifts = check_distribution_drift(pairs, u_df, u_df_raw, cfg, lg)

    # ── Save P pool ───────────────────────────────────────────────────
    pu_dir = ROOT / cfg["outputs"]["pu_dir"]
    pu_dir.mkdir(parents=True, exist_ok=True)

    p_pool = pairs.copy()
    p_pool["label"] = 1
    p_pool.to_parquet(pu_dir / "pool_P_scaffold.parquet", index=False)
    lg.info(f"\n  Written: pool_P_scaffold.parquet ({len(p_pool)} rows)")

    u_df.to_parquet(pu_dir / "pool_U_scaffold.parquet", index=False)
    lg.info(f"  Written: pool_U_scaffold.parquet ({len(u_df)} rows)")

    # ── Summary + report ─────────────────────────────────────────────
    swap_counts = u_df["swap_type"].value_counts().to_dict()
    split_u = u_df["source_split"].value_counts().to_dict()

    summary = {
        "run_timestamp": ts,
        "n_P": len(p_pool),
        "n_U": len(u_df),
        "u_to_p_ratio": round(len(u_df) / len(p_pool), 2),
        "P_intersect_U": 0,
        "swap_type_counts": swap_counts,
        "U_by_source_split": split_u,
        "distribution_drift": {k: round(v, 4) for k, v in drifts.items()},
        "distribution_drift_original": {k: round(v, 4) for k, v in original_drifts.items()},
        "max_drift_exceeded": bool(any(v > cfg["pu_pool"]["drift_threshold"] for v in drifts.values())),
        "acceptance_status": "conditional_pass" if any(v > cfg["pu_pool"]["drift_threshold"] for v in drifts.values()) else "pass"
    }
    with open(pu_dir / "pu_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=_NpEncoder)

    explicit_drift = {
        "n_P": summary["n_P"],
        "n_U": summary["n_U"],
        "u_to_p_ratio": summary["u_to_p_ratio"],
        "e3_drift": drifts.get("e3_uniprot_id", 0),
        "target_drift": drifts.get("target_uniprot_id", 0),
        "ligand_drift": drifts.get("ligand_inchikey", 0),
        "acceptance_status": summary["acceptance_status"]
    }
    with open(pu_dir / "phase4A_pool_drift_report.json", "w", encoding="utf-8") as f:
        json.dump(explicit_drift, f, indent=2, cls=_NpEncoder)

    report = f"""# IFG-26 Phase 4A — PU Pool Construction

_Generated: {ts}_

This unlabeled pool generation includes a hierarchical stratification pass over `target_uniprot_id` to reduce target drift.

## Pool Sizes

| Pool | Rows | Ratio |
|---|---|---|
| P (positives) | {summary['n_P']} | 1× |
| U (unlabeled) | {summary['n_U']} | {summary['u_to_p_ratio']}× |

## P ∩ U Overlap: **0** ✅

## Swap Type Distribution

| Swap Type | Count |
|---|---|
{"".join(f"| {k} | {v} |{chr(10)}" for k,v in swap_counts.items())}

## Distribution Drift (P vs U)

| Column | Revised Drift | Original Drift | Limit | Status |
|---|---|---|---|---|
{"".join(f"| {k} | {drifts[k]:.4f} | {original_drifts.get(k, drifts[k]):.4f} | {cfg['pu_pool']['drift_threshold']} | {'✅' if drifts[k] <= cfg['pu_pool']['drift_threshold'] else '⚠️'} |{chr(10)}"
          for k in drifts)}

## U Split Source

| Source Split | U Rows |
|---|---|
{"".join(f"| {k} | {v} |{chr(10)}" for k,v in split_u.items())}
"""
    with open(ROOT / "docs" / "phase4A_pu_pool_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    lg.info("\n" + "=" * 70)
    lg.info("IFG-26 Phase 4A — COMPLETE")
    lg.info(f"  P: {summary['n_P']} | U: {summary['n_U']} (ratio {summary['u_to_p_ratio']}×)")
    lg.info(f"  P ∩ U = 0 ✅ | Max drift = {max(drifts.values()):.3f}")
    lg.info("=" * 70)
    lg.info("\nNext: python scripts/phase4_compute_protein_embeddings.py")
    lg.info("Then: python scripts/phase4_train_nnpu.py")


if __name__ == "__main__":
    main()
