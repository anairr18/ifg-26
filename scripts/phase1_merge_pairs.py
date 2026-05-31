"""
phase1_merge_pairs.py
=====================
IFG-26 Phase 1B — Merged Pair Table Construction.

Pre-flight: verifies SHA-256 of frozen artifacts before any merge.
Aborts on mismatch.

Inputs:
  dataset/phase1/mgdb_compounds_canonicalized.csv
  dataset/phase1/mgtbind_compounds_canonicalized.csv
  data/raw/mgdb/mgdb_bioactivity.csv
  data/raw/mgtbind/mgtbind_complexes.csv
  dataset/endpoint_taxonomy.csv
  dataset/protein_mapping_manual_review.csv

Outputs (in dataset/phase1/):
  ifg26_training_pairs.csv
  phase1_merge_audit.json
  docs/phase1_merge_report.md

Dedup key:
  (compound_inchi_key, target_uniprot, e3_uniprot, mutation_flag, endpoint_tier)

Usage:
  python scripts/phase1_merge_pairs.py
"""

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

warnings.filterwarnings("ignore")


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy int64/float64 to native Python types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Frozen SHA-256 registry (from dataset_contract_v1.md)
# ---------------------------------------------------------------------------
FROZEN_SHAS = {
    "dataset/protein_mapping_manual_review.csv":
        "010e7c3c4248349e0727bccfb3d9605830b6af77eba0ce1ebfc7c9df3876ad65",
    "docs/task_definition.md":
        "faf8af36e6a8ca6fd910614bd6b53e9bc321fbce042af9c2cfaa5f5327fd4d14",
    "docs/dataset_contract_v1.md":
        None,  # Contract itself — SHA checked for existence only
}

# Tier map
TIER_MAP = {
    "degradation_evidence":   1,
    "cooperativity_evidence": 1,
    "binding_evidence":       2,
    "cellular_phenotype":     3,
    "other":                  99,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase1_merge")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(log_dir / "phase1_merge.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pre-flight SHA check
# ---------------------------------------------------------------------------
def preflight_sha_check(logger: logging.Logger) -> bool:
    """Verify frozen artifact SHAs. Returns True if all pass, False = abort."""
    logger.info("--- Pre-flight SHA verification ---")
    all_ok = True

    # Check dataset contract exists
    contract = ROOT / "docs/dataset_contract_v1.md"
    if not contract.exists():
        logger.error("[ABORT] docs/dataset_contract_v1.md not found. "
                     "Run dataset contract generation first.")
        return False
    logger.info(f"  [OK] dataset_contract_v1.md exists")

    for rel_path, expected_sha in FROZEN_SHAS.items():
        if expected_sha is None:
            continue  # existence-only check
        fpath = ROOT / rel_path
        if not fpath.exists():
            logger.error(f"  [ABORT] Frozen artifact missing: {rel_path}")
            all_ok = False
            continue
        actual = sha256_file(fpath)
        if actual == expected_sha:
            logger.info(f"  [OK] {rel_path}  SHA match ✅")
        else:
            logger.error(
                f"  [ABORT] SHA MISMATCH: {rel_path}\n"
                f"    expected: {expected_sha}\n"
                f"    actual:   {actual}\n"
                f"  File has been modified after Gate approval. "
                f"Re-run Gate review before proceeding."
            )
            all_ok = False

    if not all_ok:
        logger.error(
            "\n[ABORTED] Pre-flight SHA check failed. "
            "Do not proceed. Review remediation notes above."
        )
    else:
        logger.info("  All SHA checks PASSED — proceeding with merge.\n")

    return all_ok


# ---------------------------------------------------------------------------
# Protein mapping lookup
# ---------------------------------------------------------------------------
def build_protein_lookup(pm: pd.DataFrame) -> dict:
    """
    Returns lookup: key (uppercase raw or gene symbol) ->
      {uniprot_id, gene_symbol, is_family_level, mutation,
       exclusion_reason, mapping_assumption_flag}
    Excluded entries (exclusion_reason non-empty) are NOT added.
    """
    lookup = {}
    for _, row in pm.iterrows():
        excl = str(row.get("exclusion_reason", "")).strip()
        if excl and excl.lower() not in ("", "nan"):
            continue
        uid = str(row.get("final_uniprot_id", "")).strip()
        if not uid or uid.lower() in ("", "nan"):
            continue
        rec = {
            "uniprot_id":           uid,
            "gene_symbol":          str(row.get("gene_symbol", "")).strip(),
            "is_family_level":      str(row.get("is_family_level", "FALSE")).upper() == "TRUE",
            "mutation":             str(row.get("mutation", "")).strip(),
            "mapping_assumption_flag": str(row.get("mapping_assumption_flag", "")).strip(),
        }
        raw = str(row.get("raw_protein_string", "")).split("[SPLIT")[0].strip().upper()
        sym = rec["gene_symbol"].upper()
        for key in [raw, sym]:
            if key:
                lookup[key] = rec
    return lookup


def resolve_protein(raw: str, lookup: dict) -> dict:
    if not raw or not isinstance(raw, str):
        return {}
    k = raw.strip().upper()
    if k in lookup:
        return lookup[k]
    # partial substring match on gene symbol
    for sym, rec in lookup.items():
        if sym and sym in k:
            return rec
    return {}


# ---------------------------------------------------------------------------
# Exclusion counter
# ---------------------------------------------------------------------------
class ExclusionLedger:
    """Never drop rows silently — every exclusion is counted and categorised."""
    def __init__(self):
        self.counts: dict[str, int] = defaultdict(int)

    def record(self, reason: str, n: int = 1):
        self.counts[reason] += n

    def report(self) -> dict:
        return dict(self.counts)

    def total(self) -> int:
        return sum(self.counts.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 70)
    logger.info(f"IFG-26 Phase 1B — Merge Pair Table Construction  {ts}")
    logger.info("=" * 70)

    # ── Pre-flight ────────────────────────────────────────────────────
    if not preflight_sha_check(logger):
        sys.exit(1)

    ledger = ExclusionLedger()
    audit: dict = {
        "run_timestamp": ts,
        "sha_checks": {k: "PASS" for k in FROZEN_SHAS if FROZEN_SHAS[k]},
        "exclusions": {},
        "distributions": {},
    }

    # ── Load shared references ────────────────────────────────────────
    logger.info("Loading reference tables...")
    pm = pd.read_csv(ROOT / "dataset/protein_mapping_manual_review.csv",
                     encoding="utf-8")
    ep = pd.read_csv(ROOT / "dataset/endpoint_taxonomy.csv",
                     encoding="utf-8")
    protein_lookup = build_protein_lookup(pm)
    logger.info(f"  Protein lookup: {len(protein_lookup)} active entries "
                f"({pm['exclusion_reason'].notna().sum()} excluded entries skipped)")

    # Build endpoint → category map from taxonomy
    ep_cat_map: dict[str, str] = {}
    for _, row in ep.iterrows():
        ep_cat_map[str(row.get("raw_endpoint", "")).strip()] = row.get("taxonomy_category", "other")

    # ── Load Phase 1A canonicalized compounds ─────────────────────────
    logger.info("Loading Phase 1A canonicalized compounds...")
    mgdb_can = pd.read_csv(
        ROOT / "dataset/phase1/mgdb_compounds_canonicalized.csv",
        encoding="utf-8",
    )
    mgt_can = pd.read_csv(
        ROOT / "dataset/phase1/mgtbind_compounds_canonicalized.csv",
        encoding="utf-8",
    )

    # Only use successfully canonicalized rows
    mgdb_can_ok = mgdb_can[mgdb_can["canonicalized_ok"] == True].copy()
    mgt_can_ok  = mgt_can[mgt_can["canonicalized_ok"] == True].copy()
    n_mgdb_fail = len(mgdb_can) - len(mgdb_can_ok)
    n_mgt_fail  = len(mgt_can) - len(mgt_can_ok)
    ledger.record("mgdb_chem_fail_excluded", n_mgdb_fail)
    ledger.record("mgtbind_chem_fail_excluded", n_mgt_fail)
    logger.info(f"  MGDB canonicalized OK: {len(mgdb_can_ok)} "
                f"(excluded {n_mgdb_fail} chem failures)")
    logger.info(f"  MGTbind canonicalized OK: {len(mgt_can_ok)} "
                f"(excluded {n_mgt_fail} chem failures)")

    # MGDB SMILES lookup: source_id (MG-XXX) → {inchi_key, smiles}
    mgdb_smiles = (
        mgdb_can_ok.set_index("source_id")[["inchi_key", "canonical_isomeric_smiles",
                                             "canonical_smiles", "stereo_status"]]
    )

    # MGTbind SMILES lookup: source_id (compound integer ID)
    mgt_can_ok["source_id_str"] = mgt_can_ok["source_id"].astype(str)
    mgt_smiles = (
        mgt_can_ok.set_index("source_id_str")[["inchi_key", "canonical_isomeric_smiles",
                                                "canonical_smiles", "stereo_status"]]
    )

    # ── BUILD TABLE 1: MGTbind pairs ──────────────────────────────────
    logger.info("\nBuilding MGTbind ternary pairs...")
    mgt_cx = pd.read_csv(
        ROOT / "data/raw/mgtbind/mgtbind_complexes.csv",
        low_memory=False,
    )
    logger.info(f"  mgtbind_complexes: {len(mgt_cx)} rows")

    mgt_cx["compound_id_str"] = mgt_cx["compound_id"].astype(str)
    mgt_cx = mgt_cx.join(mgt_smiles, on="compound_id_str", how="left")

    n_mgt_no_smiles = mgt_cx["inchi_key"].isna().sum()
    ledger.record("mgtbind_no_canonical_smiles", n_mgt_no_smiles)
    logger.info(f"  Rows without canonical SMILES: {n_mgt_no_smiles} "
                f"({100*n_mgt_no_smiles/len(mgt_cx):.1f}%)")

    # Structure: protein_a = E3 binder / "recruiter"; protein_b = neo-substrate "target"
    # This is the mgtbind database convention (confirmed by kd_a/kd_b semantics)
    mgt_pairs_rows = []
    for _, row in mgt_cx.iterrows():
        ik = row.get("inchi_key")
        if not ik or pd.isna(ik):
            continue  # already counted in ledger

        moa = str(row.get("moa_type", "unknown"))
        has_pdb = bool(row.get("pdb_id") and not pd.isna(row.get("pdb_id")))

        # Mutation flag: check both protein_a and protein_b mutation columns
        mut_a = str(row.get("protein_a_mutation", "")).strip()
        mut_b = str(row.get("protein_b_mutation", "")).strip()
        mut_a = "" if mut_a.lower() in ("nan", "") else mut_a
        mut_b = "" if mut_b.lower() in ("nan", "") else mut_b
        mutation_flag = bool(mut_a or mut_b)
        mutation_detail = "|".join(filter(None, [mut_a, mut_b]))

        mgt_pairs_rows.append({
            "compound_inchi_key":          ik,
            "canonical_isomeric_smiles":   row.get("canonical_isomeric_smiles"),
            "canonical_smiles":            row.get("canonical_smiles"),
            "stereo_status":               row.get("stereo_status", "none"),
            "source_dataset":              "mgtbind",
            "source_record_id":            str(row.get("id", "")),
            "ligand_source_id":            str(row.get("compound_id", "")),
            "e3_uniprot":                  str(row.get("protein_a_uniprot_id", "")),
            "e3_gene_symbol":              str(row.get("protein_a_name", "")),
            "target_uniprot":              str(row.get("protein_b_uniprot_id", "")),
            "target_gene_symbol":          str(row.get("protein_b_name", "")),
            "moa_type":                    moa,
            "endpoint_tier":               2,
            "endpoint_category":           "binding_evidence",
            "assay_type":                  str(row.get("ternary_binding_assay", "")),
            "mutation_flag":               mutation_flag,
            "mutation_detail":             mutation_detail,
            "diagnostic_only":             False,
            "is_family_level":             False,
            "has_pdb_structure":           has_pdb,
            "pdb_id":                      str(row.get("pdb_id", "")) if has_pdb else "",
            "dc50":                        row.get("dc50"),
            "dmax":                        row.get("dmax"),
            "mapping_assumption_flag":     "",
        })

    mgt_pairs = pd.DataFrame(mgt_pairs_rows)
    logger.info(f"  MGTbind pairs built: {len(mgt_pairs)}")

    # ── BUILD TABLE 2: MGDB bioactivity pairs ────────────────────────
    logger.info("\nBuilding MGDB bioactivity pairs...")
    mgdb_bio = pd.read_csv(
        ROOT / "data/raw/mgdb/mgdb_bioactivity.csv",
        encoding="utf-8-sig", low_memory=False,
    )
    logger.info(f"  mgdb_bioactivity: {len(mgdb_bio)} rows")

    # Join canonical SMILES by compound ID
    mgdb_bio = mgdb_bio.join(mgdb_smiles, on="ID", how="left")
    n_bio_no_smiles = mgdb_bio["inchi_key"].isna().sum()
    ledger.record("mgdb_bio_no_canonical_smiles", n_bio_no_smiles)
    logger.info(f"  Rows without canonical SMILES: {n_bio_no_smiles} "
                f"({100*n_bio_no_smiles/len(mgdb_bio):.1f}%)")

    mgdb_pairs_rows = []
    n_unmapped_protein = 0
    n_excluded_species = 0

    for _, row in mgdb_bio.iterrows():
        ik = row.get("inchi_key")
        if not ik or pd.isna(ik):
            continue

        # Map endpoint to tier
        endpt_cat = "other"
        for col in ["Efficacy Type", "Assay", "Assay Method"]:
            val = str(row.get(col, "")).strip()
            if val and val.lower() not in ("nan", ""):
                cat = ep_cat_map.get(val)
                if cat:
                    endpt_cat = cat
                    break
        tier = TIER_MAP.get(endpt_cat, 99)

        # Resolve target protein
        target_raw = str(row.get("Target", "")).strip()
        pmap = resolve_protein(target_raw, protein_lookup)

        if not pmap:
            # Check if this looks like a non-human entry (heuristic)
            low = target_raw.lower()
            if any(x in low for x in ["brassica", "cucumis", "pea", "arabidopsis",
                                        "zebrafish", "mouse", "rat ", "yeast"]):
                n_excluded_species += 1
                ledger.record("excluded_non_human_target", 1)
                continue
            n_unmapped_protein += 1
            ledger.record("mgdb_protein_unmapped", 1)
            # Still include but flag as unmapped
            pmap = {}

        # Mutation flag from protein mapping
        mut = pmap.get("mutation", "")
        mut_flag = bool(mut and mut.lower() not in ("", "nan"))

        # Diagnostic only for Tier 3 and excluded for Tier 99
        if tier == 99:
            ledger.record("excluded_other_endpoint", 1)
            continue  # hard exclude

        diagnostic_only = (tier == 3)

        mgdb_pairs_rows.append({
            "compound_inchi_key":          ik,
            "canonical_isomeric_smiles":   row.get("canonical_isomeric_smiles"),
            "canonical_smiles":            row.get("canonical_smiles"),
            "stereo_status":               row.get("stereo_status", "none"),
            "source_dataset":              "mgdb",
            "source_record_id":            str(row.get("ID", "")),
            "ligand_source_id":            str(row.get("ID", "")),
            "e3_uniprot":                  "",   # MGDB bioactivity doesn't split E3/neo
            "e3_gene_symbol":              "",
            "target_uniprot":              pmap.get("uniprot_id", ""),
            "target_gene_symbol":          pmap.get("gene_symbol", target_raw[:60]),
            "moa_type":                    "unknown",
            "endpoint_tier":               tier,
            "endpoint_category":           endpt_cat,
            "assay_type":                  str(row.get("Assay Method", "")),
            "mutation_flag":               mut_flag,
            "mutation_detail":             mut if mut_flag else "",
            "diagnostic_only":             diagnostic_only,
            "is_family_level":             pmap.get("is_family_level", False),
            "has_pdb_structure":           False,
            "pdb_id":                      "",
            "dc50":                        None,
            "dmax":                        None,
            "mapping_assumption_flag":     pmap.get("mapping_assumption_flag", ""),
        })

    mgdb_pairs = pd.DataFrame(mgdb_pairs_rows)
    logger.info(f"  MGDB pairs built: {len(mgdb_pairs)} "
                f"(unmapped protein: {n_unmapped_protein}, "
                f"non-human excluded: {n_excluded_species})")

    # ── COMBINE ───────────────────────────────────────────────────────
    logger.info("\nCombining all pairs...")
    combined = pd.concat([mgt_pairs, mgdb_pairs], ignore_index=True)
    n_pre_dedup = len(combined)
    logger.info(f"  Pre-dedup total: {n_pre_dedup}")

    # ── DEDUP ─────────────────────────────────────────────────────────
    logger.info("Deduplicating...")
    DEDUP_KEY = ["compound_inchi_key", "target_uniprot", "e3_uniprot",
                 "mutation_flag", "endpoint_tier"]

    # Sort by tier ascending (Tier 1 kept over Tier 2 over Tier 3)
    combined_sorted = combined.sort_values("endpoint_tier", ascending=True)
    deduped = combined_sorted.drop_duplicates(subset=DEDUP_KEY, keep="first").copy()

    n_dupes = n_pre_dedup - len(deduped)
    ledger.record("exact_duplicates_removed", n_dupes)
    logger.info(f"  Post-dedup: {len(deduped)} "
                f"(removed {n_dupes} exact duplicates on {DEDUP_KEY})")

    # Cross-dataset overlap
    iki_mgt  = set(mgt_pairs["compound_inchi_key"].dropna())
    iki_mgdb = set(mgdb_pairs["compound_inchi_key"].dropna())
    cross_overlap = len(iki_mgt & iki_mgdb)
    logger.info(f"  Cross-dataset InChIKey overlap: {cross_overlap} compounds")
    deduped["in_both_datasets"] = deduped["compound_inchi_key"].isin(iki_mgt & iki_mgdb)

    # ── SPLIT: training vs diagnostic ────────────────────────────────
    training = deduped[deduped["endpoint_tier"].isin([1, 2]) &
                       (~deduped["diagnostic_only"])].copy()
    diagnostic = deduped[deduped["diagnostic_only"] == True].copy()
    excluded_family = deduped[deduped["is_family_level"] == True]
    n_family = len(excluded_family)
    ledger.record("family_level_in_training", n_family)

    logger.info(f"\n  Training pairs (Tier 1+2, diagnostic_only=False): {len(training)}")
    logger.info(f"  Diagnostic-only (Tier 3): {len(diagnostic)}")
    logger.info(f"  Family-level entries in training: {n_family}")

    # ── OUTPUTS ───────────────────────────────────────────────────────
    out_base = ROOT / "dataset/phase1"
    out_base.mkdir(parents=True, exist_ok=True)

    pairs_path = out_base / "ifg26_training_pairs.csv"
    diag_path  = out_base / "ifg26_diagnostic_pairs.csv"
    deduped.to_csv(pairs_path, index=False, encoding="utf-8")
    diagnostic.to_csv(diag_path, index=False, encoding="utf-8")
    logger.info(f"\n  Written: {pairs_path}")
    logger.info(f"  Written: {diag_path}")

    # ── EMBEDDED AUDIT ────────────────────────────────────────────────
    logger.info("\n--- Embedded Audit ---")

    # Compounds per E3
    e3_counts = (
        training[training["e3_uniprot"] != ""]
        .groupby("e3_uniprot")["compound_inchi_key"].nunique()
        .sort_values(ascending=False)
    )
    logger.info(f"  E3s with compound counts (top 10):")
    for uid, cnt in e3_counts.head(10).items():
        logger.info(f"    {uid}: {cnt} compounds")

    # Targets per E3
    targets_per_e3 = (
        training[training["e3_uniprot"] != ""]
        .groupby("e3_uniprot")["target_uniprot"].nunique()
        .sort_values(ascending=False)
    )

    # Mutation frequencies
    n_mutation = training["mutation_flag"].sum()
    unique_mutations = training[training["mutation_flag"]]["mutation_detail"].value_counts().to_dict()

    # Tier counts
    tier_counts = training["endpoint_tier"].value_counts().to_dict()
    cat_counts  = training["endpoint_category"].value_counts().to_dict()
    src_counts  = training["source_dataset"].value_counts().to_dict()

    # % rows lost due to missing SMILES
    total_bio_rows = len(mgdb_bio) + len(mgt_cx)
    total_no_smiles = n_bio_no_smiles + n_mgt_no_smiles
    pct_smiles_lost = 100 * total_no_smiles / total_bio_rows if total_bio_rows else 0

    logger.info(f"  Tier distribution: {tier_counts}")
    logger.info(f"  Category distribution: {cat_counts}")
    logger.info(f"  Source distribution: {src_counts}")
    logger.info(f"  Mutation-tracked: {int(n_mutation)}")
    logger.info(f"  Rows lost due to no canonical SMILES: {total_no_smiles} "
                f"({pct_smiles_lost:.1f}%)")

    # Build audit dict
    audit["totals"] = {
        "pre_dedup_pairs":       n_pre_dedup,
        "post_dedup_pairs":      len(deduped),
        "training_pairs":        len(training),
        "diagnostic_pairs":      len(diagnostic),
        "unique_compounds":      int(training["compound_inchi_key"].nunique()),
        "unique_e3s":            int(training[training["e3_uniprot"]!=""]["e3_uniprot"].nunique()),
        "unique_targets":        int(training["target_uniprot"].replace("", pd.NA).dropna().nunique()),
        "mutation_tracked_rows": int(n_mutation),
        "family_level_rows":     int(n_family),
        "cross_dataset_overlap": int(cross_overlap),
    }
    audit["tier_distribution"]     = {str(k): int(v) for k, v in tier_counts.items()}
    audit["category_distribution"] = {k: int(v) for k, v in cat_counts.items()}
    audit["source_distribution"]   = {k: int(v) for k, v in src_counts.items()}
    audit["mutation_details"]      = {k: int(v) for k, v in unique_mutations.items()}
    audit["exclusions"]            = ledger.report()
    audit["pct_rows_lost_no_smiles"] = round(pct_smiles_lost, 2)
    audit["e3_compound_counts"] = {k: int(v) for k, v in e3_counts.head(20).items()}
    audit["targets_per_e3"]     = {k: int(v) for k, v in targets_per_e3.head(20).items()}

    audit_path = out_base / "phase1_merge_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, cls=_NumpyEncoder)
    logger.info(f"\n  Written: {audit_path}")

    # ── MERGE REPORT ─────────────────────────────────────────────────
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    e3_table = "\n".join(
        f"| {uid} | {cnt} | {targets_per_e3.get(uid, 0)} |"
        for uid, cnt in e3_counts.head(15).items()
    )
    excl_table = "\n".join(
        f"| `{k}` | {v} |"
        for k, v in ledger.report().items()
    )

    report_md = f"""# IFG-26 Phase 1B — Merge Report

_Generated: {ts}_

## Pre-flight Check

| Artifact | SHA | Status |
|---|---|---|
| `protein_mapping_manual_review.csv` | `010e7c3c…` | ✅ MATCH |
| `task_definition.md` | `faf8af36…` | ✅ MATCH |
| `dataset_contract_v1.md` | — | ✅ EXISTS |

## Pair Table Summary

| Metric | Count |
|---|---|
| Pre-dedup pairs | {n_pre_dedup} |
| Post-dedup pairs | {len(deduped)} |
| **Training pairs (Tier 1+2)** | **{len(training)}** |
| Diagnostic-only (Tier 3) | {len(diagnostic)} |
| Unique ligands (InChIKey) | {training['compound_inchi_key'].nunique()} |
| Unique E3 UniProts | {training[training['e3_uniprot']!='']['e3_uniprot'].nunique()} |
| Unique target UniProts | {int(training['target_uniprot'].replace('', pd.NA).dropna().nunique())} |
| Mutation-tracked rows | {int(n_mutation)} |
| Family-level rows | {n_family} |
| Cross-dataset overlap | {cross_overlap} compounds |

## Tier Distribution

| Tier | Category | Count |
|---|---|---|
| 1 | `degradation_evidence` | {cat_counts.get('degradation_evidence', 0)} |
| 1 | `cooperativity_evidence` | {cat_counts.get('cooperativity_evidence', 0)} |
| 2 | `binding_evidence` | {cat_counts.get('binding_evidence', 0)} |

## Source Distribution

| Source | Pairs |
|---|---|
| `mgtbind` | {src_counts.get('mgtbind', 0)} |
| `mgdb` | {src_counts.get('mgdb', 0)} |

## Top E3 Ligases by Compound Count

| E3 UniProt | Compounds | Neo-substrates |
|---|---|---|
{e3_table}

## Exclusion Ledger (nothing dropped silently)

| Reason | Count |
|---|---|
{excl_table}

## Rows Lost Due to Missing Canonical SMILES

- Total rows in bioactivity/complexes: {total_bio_rows}
- No canonical SMILES: {total_no_smiles} ({pct_smiles_lost:.1f}%)

## Output Files

| File | Rows | Description |
|---|---|---|
| `ifg26_training_pairs.csv` | {len(training)} | Tier 1+2 training positives |
| `ifg26_diagnostic_pairs.csv` | {len(diagnostic)} | Tier 3 diagnostic (OOD probe) |
| `phase1_merge_audit.json` | — | Full audit with distributions |
"""
    report_path = docs_dir / "phase1_merge_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"  Written: {report_path}")

    # ── FINAL ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("IFG-26 Phase 1B Merge — COMPLETE")
    logger.info(f"  Training pairs:     {len(training)}")
    logger.info(f"  Unique ligands:     {training['compound_inchi_key'].nunique()}")
    logger.info(f"  Tier 1:             {tier_counts.get(1, 0)}")
    logger.info(f"  Tier 2:             {tier_counts.get(2, 0)}")
    logger.info(f"  Mutation-tracked:   {int(n_mutation)}")
    logger.info(f"  Total exclusions:   {ledger.total()}")
    logger.info("=" * 70)
    logger.info("\nNext step: python scripts/phase1_split.py")


if __name__ == "__main__":
    main()
