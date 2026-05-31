"""
phase3_featurize.py
===================
IFG-26 Phase 3A — Ligand + Protein Featurization + Similarity Binning.

Config:   configs/experiment/phase3_default.yaml
Usage:    python scripts/phase3_featurize.py [--config <path>]

Outputs (data/features/):
    ligand_index.parquet          InChIKey → row index
    ligands_ecfp4.npy             (N_ligands × 2048) uint8 bit vectors
    ligands_physchem.parquet      (N_ligands × 13) float
    protein_index.parquet         uniprot_id → label encoding
    protein_embeddings.parquet    Tier P1: one-hot label; P2 stub note
    scaffold_pairs_index.parquet  row-level pair table with feature pointers
    test_similarity_bins.parquet  per-test-ligand NN tanimoto + bin label

Outputs (results/figures/phase3/):
    test_nn_tanimoto_hist.png
    test_nn_bin_counts.png
"""

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
except ImportError as e:
    print(f"[FATAL] RDKit not available: {e}")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase3_default.yaml"

# ---------------------------------------------------------------------------
# Fixed physchem descriptor functions — order must never change
# ---------------------------------------------------------------------------
PHYSCHEM_FUNCS = {
    "MolWt":             Descriptors.MolWt,
    "MolLogP":           Descriptors.MolLogP,
    "NumHDonors":        rdMolDescriptors.CalcNumHBD,
    "NumHAcceptors":     rdMolDescriptors.CalcNumHBA,
    "TPSA":              rdMolDescriptors.CalcTPSA,
    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount":         rdMolDescriptors.CalcNumRings,
    "FormalCharge":      Chem.GetFormalCharge,
    "FractionCSP3":      rdMolDescriptors.CalcFractionCSP3,
    "NumAromaticRings":  rdMolDescriptors.CalcNumAromaticRings,
    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "NumHeteroatoms":    rdMolDescriptors.CalcNumHeteroatoms,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def setup_logging(log_dir: Path, name="phase3_featurize") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def abort(logger, msg):
    logger.error(f"[ABORTED] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight(cfg, logger):
    logger.info("--- Pre-flight SHA verification ---")
    for rel, expected in cfg.get("preflight", {}).get("frozen_shas", {}).items():
        path = ROOT / rel
        if not path.exists():
            abort(logger, f"Missing: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            abort(logger, f"SHA MISMATCH: {rel}\n  expected: {expected}\n  actual: {actual}")
        logger.info(f"  [OK] {rel}")
    logger.info("  All SHA checks PASSED.\n")


# ---------------------------------------------------------------------------
# Ligand features
# ---------------------------------------------------------------------------
from rdkit.Chem import rdFingerprintGenerator

def compute_ecfp4(smiles: str, radius: int, nbits: int) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles) if smiles and isinstance(smiles, str) else None
    if mol is None:
        return None
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fp = mfpgen.GetFingerprint(mol)
    arr = np.zeros(nbits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_physchem(smiles: str, desc_names: list) -> dict | None:
    mol = Chem.MolFromSmiles(smiles) if smiles and isinstance(smiles, str) else None
    if mol is None:
        return None
    row = {}
    for name in desc_names:
        fn = PHYSCHEM_FUNCS.get(name)
        try:
            row[name] = float(fn(mol)) if fn else float("nan")
        except Exception:
            row[name] = float("nan")
    return row


def featurize_ligands(ik_smiles: dict, cfg: dict, logger: logging.Logger,
                      features_dir: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Returns (ligand_index_df, ecfp4_matrix, physchem_df).
    Caches to disk; reloads if cache exists.
    """
    ecfp_path   = features_dir / "ligands_ecfp4.npy"
    phys_path   = features_dir / "ligands_physchem.parquet"
    idx_path    = features_dir / "ligand_index.parquet"

    if ecfp_path.exists() and phys_path.exists() and idx_path.exists():
        logger.info("  Feature cache hit — loading from disk...")
        lig_idx  = pd.read_parquet(idx_path)
        ecfp_mat = np.load(str(ecfp_path))
        phys_df  = pd.read_parquet(phys_path)
        logger.info(f"  Loaded {len(lig_idx)} ligands from cache.")
        return lig_idx, ecfp_mat, phys_df

    ecfp_cfg = cfg.get("ligand_features", {}).get("ecfp4", {})
    radius   = ecfp_cfg.get("radius", 2)
    nbits    = ecfp_cfg.get("nbits", 2048)
    desc_names = cfg.get("ligand_features", {}).get("physchem", {}).get("descriptors",
                     list(PHYSCHEM_FUNCS.keys()))

    ik_list = sorted(ik_smiles.keys())
    n = len(ik_list)
    logger.info(f"  Computing ECFP4 + physchem for {n} unique InChIKeys...")

    ecfp_mat  = np.zeros((n, nbits), dtype=np.uint8)
    phys_rows = []
    fail_ecfp = 0
    fail_phys = 0
    t0 = time.perf_counter()

    for i, ik in enumerate(ik_list):
        smi = ik_smiles[ik]
        fp = compute_ecfp4(smi, radius, nbits)
        if fp is not None:
            ecfp_mat[i] = fp
        else:
            fail_ecfp += 1

        pc = compute_physchem(smi, desc_names)
        if pc is not None:
            phys_rows.append({"inchi_key": ik, **pc})
        else:
            phys_rows.append({"inchi_key": ik, **{d: float("nan") for d in desc_names}})
            fail_phys += 1

        if (i + 1) % 1000 == 0:
            el = time.perf_counter() - t0
            rate = (i + 1) / el
            logger.info(f"    {i+1}/{n} | {rate:.0f} mol/s | ETA {(n-i-1)/rate:.0f}s")

    lig_idx = pd.DataFrame({"inchi_key": ik_list,
                             "row_idx": list(range(n))})
    phys_df = pd.DataFrame(phys_rows)

    logger.info(f"  Done: {n-fail_ecfp} ECFP4 OK, {fail_ecfp} fail | "
                f"{n-fail_phys} physchem OK, {fail_phys} fail")

    # Save
    np.save(str(ecfp_path), ecfp_mat)
    phys_df.to_parquet(phys_path, index=False)
    lig_idx.to_parquet(idx_path, index=False)
    logger.info(f"  Saved: {ecfp_path.name}, {phys_path.name}, {idx_path.name}")

    return lig_idx, ecfp_mat, phys_df


# ---------------------------------------------------------------------------
# Protein features
# ---------------------------------------------------------------------------
def featurize_proteins(all_uniprots: list, cfg: dict, logger: logging.Logger,
                       features_dir: Path) -> pd.DataFrame:
    """Tier P1: label-encoding. Returns protein_index_df."""
    p_idx_path = features_dir / "protein_index.parquet"
    p_emb_path = features_dir / "protein_embeddings.parquet"

    if p_idx_path.exists() and p_emb_path.exists():
        logger.info("  Protein feature cache hit — loading...")
        return pd.read_parquet(p_idx_path)

    logger.info(f"  Encoding {len(all_uniprots)} unique UniProt IDs (Tier P1)...")
    sorted_uids = sorted(set(u for u in all_uniprots if u and u not in ("", "nan")))
    p_idx = pd.DataFrame({
        "uniprot_id": sorted_uids,
        "label_id":   list(range(len(sorted_uids))),
    })
    p_idx.to_parquet(p_idx_path, index=False)

    # Tier P2 stub — document that ESM2/ProtT5 not available
    tier_p2_note = cfg.get("protein_features", {}).get("tier_p2", {}).get(
        "note", "ESM2/ProtT5 embedding deferred to Phase 4."
    )
    emb_stub = pd.DataFrame({
        "uniprot_id": sorted_uids,
        "label_id":   list(range(len(sorted_uids))),
        "embedding_available": False,
        "embedding_method": "STUB",
        "note": tier_p2_note,
    })
    emb_stub.to_parquet(p_emb_path, index=False)

    logger.info(f"  Protein index: {len(p_idx)} IDs | P2 stub written.")
    logger.info(f"  Saved: {p_idx_path.name}, {p_emb_path.name}")
    return p_idx


# ---------------------------------------------------------------------------
# Scaffold pair table
# ---------------------------------------------------------------------------
def build_pair_index(train_df, val_df, test_df, lig_idx, p_idx, features_dir, logger):
    """Assemble scaffold_pairs_index.parquet with feature row pointers."""
    ik_to_row  = lig_idx.set_index("inchi_key")["row_idx"].to_dict()
    uid_to_lbl = p_idx.set_index("uniprot_id")["label_id"].to_dict()

    dfs = [("train", train_df), ("val", val_df), ("test", test_df)]
    rows = []
    for split_name, df in dfs:
        for _, row in df.iterrows():
            ik  = row["compound_inchi_key"]
            e3  = str(row.get("e3_uniprot", ""))
            tgt = str(row.get("target_uniprot", ""))
            rows.append({
                "split":                 split_name,
                "ligand_inchikey":       ik,
                "ligand_feature_row":    ik_to_row.get(ik, -1),
                "e3_uniprot_id":         e3,
                "e3_label_id":           uid_to_lbl.get(e3, -1),
                "target_uniprot_id":     tgt,
                "target_label_id":       uid_to_lbl.get(tgt, -1),
                "label":                 1,
                "tier":                  int(row.get("endpoint_tier", -1)),
                "endpoint_category":     row.get("endpoint_category", ""),
                "source_dataset":        row.get("source_dataset", ""),
                "mutation_flag":         bool(row.get("mutation_flag", False)),
                "mutation_detail":       str(row.get("mutation_detail", "")),
                "is_family_level":       bool(row.get("is_family_level", False)),
                "has_pdb_structure":     bool(row.get("has_pdb_structure", False)),
            })

    pair_df = pd.DataFrame(rows)
    out = features_dir / "scaffold_pairs_index.parquet"
    pair_df.to_parquet(out, index=False)
    logger.info(f"  Pair index: {len(pair_df)} rows → {out.name}")
    return pair_df


# ---------------------------------------------------------------------------
# Similarity binning
# ---------------------------------------------------------------------------
def compute_sim_bins(train_df, test_df, lig_idx, ecfp_mat, cfg, figures_dir, logger):
    """Per test-ligand NN Tanimoto + bin assignment. Returns bin_df."""
    bin_path = Path(cfg["outputs"]["features_dir"].replace("data/features",
                    str(ROOT / "data/features"))) / "test_similarity_bins.parquet"
    # Use absolute path directly
    feat_dir = ROOT / cfg["outputs"]["features_dir"]
    bin_path  = feat_dir / "test_similarity_bins.parquet"

    thresholds = cfg.get("similarity_bins", {}).get("thresholds", [0.60, 0.80, 0.85, 0.95])
    ik_to_row  = lig_idx.set_index("inchi_key")["row_idx"].to_dict()

    train_iks = train_df["compound_inchi_key"].unique().tolist()
    test_iks  = test_df["compound_inchi_key"].unique().tolist()

    logger.info(f"  Building similarity bins: {len(test_iks)} test vs {len(train_iks)} train IKs...")

    # Build RDKit fingerprint objects for Tanimoto
    from rdkit.Chem import DataStructs
    from rdkit.DataStructs import ExplicitBitVect

    def row_to_fp(ik):
        r = ik_to_row.get(ik, -1)
        if r == -1:
            return None
        arr = ecfp_mat[r].tolist()
        fp  = DataStructs.ExplicitBitVect(len(arr))
        for j, b in enumerate(arr):
            if b:
                fp.SetBit(j)
        return fp

    train_fps = [(ik, row_to_fp(ik)) for ik in train_iks]
    train_fps = [(ik, fp) for ik, fp in train_fps if fp is not None]
    train_fp_list = [fp for _, fp in train_fps]

    bin_rows = []
    labels = ["A", "B", "C", "D", "E"]   # A: [0,t0), B: [t0,t1), ..., E: [t4,1]

    for ik in test_iks:
        fp = row_to_fp(ik)
        if fp is None:
            nn_sim = 0.0
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fp_list)
            nn_sim = max(sims) if sims else 0.0

        # Bin assignment
        bin_label = labels[-1]  # default to E (highest)
        for i, t in enumerate(thresholds):
            if nn_sim < t:
                bin_label = labels[i]
                break

        bin_rows.append({
            "compound_inchi_key": ik,
            "nn_tanimoto_to_train": nn_sim,
            "similarity_bin": bin_label,
        })

    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_parquet(bin_path, index=False)
    logger.info(f"  Bins: {bin_df['similarity_bin'].value_counts().to_dict()}")
    logger.info(f"  Saved: {bin_path.name}")

    # Plots
    figures_dir.mkdir(parents=True, exist_ok=True)
    nn_vals = bin_df["nn_tanimoto_to_train"].values
    hist_bins = cfg.get("similarity_bins", {}).get("hist_bins", 50)

    # Histogram
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(nn_vals, bins=hist_bins, color="#1565C0", edgecolor="white", linewidth=0.4)
    for t, col in zip(thresholds, ["#FF7043", "#FFA726", "#66BB6A", "#AB47BC"]):
        ax.axvline(t, color=col, linestyle="--", linewidth=1.2,
                   label=f"t={t}")
    ax.set_xlabel("Max Tanimoto (ECFP4, test → train)", fontsize=11)
    ax.set_ylabel("Test ligands", fontsize=11)
    ax.set_title("IFG-26 Test Set NN Tanimoto Distribution (Scaffold Split)", fontsize=12)
    ax.legend(fontsize=8, title="Bin edges")
    plt.tight_layout()
    fig.savefig(figures_dir / "test_nn_tanimoto_hist.png", dpi=150)
    plt.close()

    # Bar chart of bin counts
    bin_counts = bin_df["similarity_bin"].value_counts().reindex(labels).fillna(0)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colors = ["#1976D2", "#43A047", "#F9A825", "#E64A19", "#8E24AA"]
    bars = ax2.bar(labels, bin_counts.values, color=colors, edgecolor="white")
    for bar, cnt in zip(bars, bin_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{int(cnt)}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Similarity Bin (A=low, E=near-identical)", fontsize=11)
    ax2.set_ylabel("Test ligands", fontsize=11)
    ax2.set_title("Test Set Similarity Bins (ECFP4 NN to Train)", fontsize=12)
    bin_labels_text = ["A\n[0,0.60)", "B\n[0.60,0.80)", "C\n[0.80,0.85)",
                       "D\n[0.85,0.95)", "E\n[0.95,1.0]"]
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(bin_labels_text, fontsize=9)
    plt.tight_layout()
    fig2.savefig(figures_dir / "test_nn_bin_counts.png", dpi=150)
    plt.close()
    logger.info("  Saved: test_nn_tanimoto_hist.png, test_nn_bin_counts.png")

    return bin_df


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
    logger.info(f"IFG-26 Phase 3A — Featurization  {ts}")
    logger.info("=" * 70)

    preflight(cfg, logger)

    features_dir = ROOT / cfg["outputs"]["features_dir"]
    figures_dir  = ROOT / cfg["outputs"]["figures_dir"]
    features_dir.mkdir(parents=True, exist_ok=True)

    # ── Load scaffold splits ──────────────────────────────────────────
    logger.info("Loading scaffold splits...")
    train_df = pd.read_csv(ROOT / cfg["inputs"]["train_scaffold"], low_memory=False)
    val_df   = pd.read_csv(ROOT / cfg["inputs"]["val_scaffold"],   low_memory=False)
    test_df  = pd.read_csv(ROOT / cfg["inputs"]["test_scaffold"],  low_memory=False)
    logger.info(f"  train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # ── Build InChIKey → SMILES map ───────────────────────────────────
    ik_smiles = (
        all_df.drop_duplicates("compound_inchi_key")
              .set_index("compound_inchi_key")["canonical_smiles"]
              .to_dict()
    )
    logger.info(f"  Unique InChIKeys: {len(ik_smiles)}")

    # ── Ligand features ───────────────────────────────────────────────
    logger.info("\n--- Ligand Features ---")
    lig_idx, ecfp_mat, phys_df = featurize_ligands(ik_smiles, cfg, logger, features_dir)

    # ── Protein features ──────────────────────────────────────────────
    logger.info("\n--- Protein Features ---")
    all_uniprots = (
        list(all_df["e3_uniprot"].dropna().unique()) +
        list(all_df["target_uniprot"].dropna().unique())
    )
    p_idx = featurize_proteins(all_uniprots, cfg, logger, features_dir)

    # ── Scaffold pair index ───────────────────────────────────────────
    logger.info("\n--- Scaffold Pair Index ---")
    pair_df = build_pair_index(train_df, val_df, test_df, lig_idx, p_idx,
                               features_dir, logger)

    # ── Similarity bins ───────────────────────────────────────────────
    logger.info("\n--- Similarity Binning (test set) ---")
    bin_df = compute_sim_bins(train_df, test_df, lig_idx, ecfp_mat,
                              cfg, figures_dir, logger)

    # ── Summary ───────────────────────────────────────────────────────
    bin_counts = bin_df["similarity_bin"].value_counts().reindex(
        ["A","B","C","D","E"]).fillna(0).astype(int)
    n_test_unique = len(bin_df)

    summary = {
        "run_timestamp": ts,
        "n_unique_ligands": len(lig_idx),
        "ecfp4_shape": list(ecfp_mat.shape),
        "physchem_cols": [c for c in phys_df.columns if c != "inchi_key"],
        "n_unique_proteins": len(p_idx),
        "n_train_rows": len(train_df), "n_val_rows": len(val_df), "n_test_rows": len(test_df),
        "test_similarity_bins": bin_counts.to_dict(),
        "test_bin_e_pct": round(100 * int(bin_counts.get("E", 0)) / max(n_test_unique, 1), 2),
    }
    with open(features_dir / "phase3_featurize_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("IFG-26 Phase 3A Featurization — COMPLETE")
    logger.info(f"  Unique ligands: {len(lig_idx)} | ECFP4 shape: {ecfp_mat.shape}")
    logger.info(f"  physchem: {len(phys_df.columns)-1} descriptors")
    logger.info(f"  Unique proteins: {len(p_idx)}")
    logger.info(f"  Similarity bins: {bin_counts.to_dict()}")
    logger.info(f"  Bin E (≥0.95): {int(bin_counts.get('E',0))} "
                f"({summary['test_bin_e_pct']:.1f}% of test unique ligands)")
    logger.info("=" * 70)
    logger.info("\nNext step: python scripts/phase3_baseline_probes.py")


if __name__ == "__main__":
    main()
