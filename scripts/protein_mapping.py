"""
protein_mapping.py
------------------
Phase 0 — Steps 7, 8, 9: Extract unique protein names and query UniProt for mapping.

Usage:
    python scripts/protein_mapping.py [--max-candidates 3]

Outputs:
    dataset/unique_protein_names.csv
    dataset/protein_mapping_candidates.csv
    dataset/protein_mapping_manual_review.csv
    logs/mapping_log.txt

Rules:
    - Do NOT select or impose a single UniProt ID; generate candidates for human review.
    - Flag ambiguous mappings (multiple close matches, isoforms, multi-organism).
    - Use string similarity scoring for confidence.
"""

import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
DOCS_DIR = PROJECT_ROOT / "docs"
LOG_FILE = PROJECT_ROOT / "logs" / "mapping_log.txt"

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
MAX_CANDIDATES = 3
UNIPROT_RATE_LIMIT_DELAY = 0.5  # seconds between API calls
REQUEST_TIMEOUT = 30

AMBIGUITY_SCORE_DIFF_THRESHOLD = 10  # fuzzy score gap between top candidates
AMBIGUITY_MIN_SCORE = 60  # minimum fuzzy score to consider a match valid

# Columns in raw files that likely contain protein names
PROTEIN_COL_CANDIDATES = [
    "e3_ligase", "neo_substrate", "target", "protein", "protein_1",
    "protein_2", "receptor", "ligase", "substrate",
]


# ---------------------------------------------------------------------------
# Unicode protection layer
# ---------------------------------------------------------------------------

def sanitize_unicode(text: str) -> str:
    """Normalize Unicode Greek letters to ASCII before API/string matching.

    Phase 0 review found that raw protein strings containing Unicode characters
    (e.g. CK1α, 14-3-3σ, 14-3-3γ) caused UniProt API errors or null returns.
    This function must be applied to ALL protein name strings before any
    UniProt query, fuzzy match, or CSV output.

    Approved replacements (Gate 1, 2026-02-21, Biologist Reviewer):
        α → alpha  (e.g. CK1α → CK1alpha)
        β → beta
        γ → gamma  (e.g. 14-3-3γ → 14-3-3gamma)
        σ → sigma  (e.g. 14-3-3σ → 14-3-3sigma)
        ζ → zeta
    """
    return text.translate(str.maketrans({
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "σ": "sigma",
        "ζ": "zeta",
    }))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("protein_mapping")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Protein extraction
# ---------------------------------------------------------------------------

def find_col(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return all columns matching any candidate (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    found = []
    for cand in candidates:
        if cand.lower() in lower_map and lower_map[cand.lower()] not in found:
            found.append(lower_map[cand.lower()])
    return found


def extract_unique_proteins(raw_dirs: dict) -> list[str]:
    """Extract all unique protein names from raw CSV files."""
    all_names = set()
    for ds_name, raw_dir in raw_dirs.items():
        if not raw_dir.exists():
            continue
        for fpath in sorted(raw_dir.glob("*.csv")):
            try:
                df = pd.read_csv(fpath, low_memory=False)
            except Exception:
                continue
            cols = find_col(df, PROTEIN_COL_CANDIDATES)
            for col in cols:
                vals = df[col].dropna().astype(str).unique()
                all_names.update(v.strip() for v in vals if v.strip() and v.strip().lower() != "nan")
    return sorted(all_names)


# ---------------------------------------------------------------------------
# UniProt query
# ---------------------------------------------------------------------------

def query_uniprot(protein_name: str, max_results: int = 5) -> list[dict]:
    """Query UniProt REST API and return top candidate results."""
    params = {
        "query": f'protein_name:"{protein_name}" OR gene:"{protein_name}"',
        "format": "json",
        "size": max_results,
        "fields": "accession,protein_name,gene_names,organism_name,sequence",
    }
    try:
        resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
    except requests.RequestException as e:
        return [{"error": str(e)}]

    candidates = []
    for r in results:
        try:
            accession = r.get("primaryAccession", "N/A")
            prot_name = (
                r.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", "N/A")
            )
            organism = r.get("organism", {}).get("scientificName", "N/A")
            seq_len = r.get("sequence", {}).get("length", 0)
            gene_names = [
                g.get("value", "")
                for genes in r.get("genes", [])
                for g in genes.get("geneName", [{}])
            ]
            gene_str = "; ".join(gene_names[:3])
        except Exception:
            prot_name = "parse_error"
            organism = "N/A"
            seq_len = 0
            gene_str = "N/A"

        # Compute fuzzy string similarity
        score = fuzz.token_sort_ratio(protein_name.lower(), prot_name.lower())

        candidates.append({
            "uniprot_accession": accession,
            "uniprot_protein_name": prot_name,
            "gene_names": gene_str,
            "organism": organism,
            "sequence_length": seq_len,
            "fuzzy_score": score,
        })

    return sorted(candidates, key=lambda x: -x.get("fuzzy_score", 0))


def detect_ambiguity(candidates: list[dict]) -> str:
    """Classify mapping ambiguity."""
    if not candidates or candidates[0].get("error"):
        return "API_ERROR"
    top = candidates[0]
    if top.get("fuzzy_score", 0) < AMBIGUITY_MIN_SCORE:
        return "NO_CONFIDENT_MATCH"
    if len(candidates) < 2:
        return "UNAMBIGUOUS"
    score_diff = top["fuzzy_score"] - candidates[1].get("fuzzy_score", 0)
    if score_diff < AMBIGUITY_SCORE_DIFF_THRESHOLD:
        return "AMBIGUOUS_MULTIPLE_CLOSE_MATCHES"
    # Check multi-organism
    organisms = set(c.get("organism", "") for c in candidates[:3])
    if len(organisms) > 1:
        return "AMBIGUOUS_MULTI_ORGANISM"
    return "UNAMBIGUOUS"


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def build_candidate_rows(protein_name: str, candidates: list[dict], ambiguity: str) -> list[dict]:
    """Flatten candidates into table rows."""
    rows = []
    for i, c in enumerate(candidates[:MAX_CANDIDATES], 1):
        rows.append({
            "original_name": protein_name,
            "rank": i,
            "uniprot_accession": c.get("uniprot_accession", "N/A"),
            "uniprot_protein_name": c.get("uniprot_protein_name", "N/A"),
            "gene_names": c.get("gene_names", "N/A"),
            "organism": c.get("organism", "N/A"),
            "sequence_length": c.get("sequence_length", "N/A"),
            "fuzzy_score": c.get("fuzzy_score", 0),
            "ambiguity_flag": ambiguity,
            "api_error": c.get("error", ""),
        })
    # Pad to MAX_CANDIDATES rows if fewer returned
    while len(rows) < MAX_CANDIDATES:
        rows.append({
            "original_name": protein_name,
            "rank": len(rows) + 1,
            "uniprot_accession": "NO_RESULT",
            "uniprot_protein_name": "NO_RESULT",
            "gene_names": "",
            "organism": "",
            "sequence_length": 0,
            "fuzzy_score": 0,
            "ambiguity_flag": ambiguity,
            "api_error": "",
        })
    return rows


def build_manual_review_table(protein_names: list[str], candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Generate wide-format manual review template."""
    rows = []
    for name in protein_names:
        sub = candidates_df[candidates_df["original_name"] == name].sort_values("rank")
        cands = sub["uniprot_accession"].tolist()
        rows.append({
            "original_name": name,
            "candidate_uniprot_1": cands[0] if len(cands) > 0 else "",
            "candidate_uniprot_2": cands[1] if len(cands) > 1 else "",
            "candidate_uniprot_3": cands[2] if len(cands) > 2 else "",
            "ambiguity_flag": sub["ambiguity_flag"].iloc[0] if not sub.empty else "UNKNOWN",
            "chosen_uniprot": "",      # to be filled by human reviewer
            "reviewer_notes": "",
            "reviewer_initials": "",
            "review_date": "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Phase 0 — Protein Mapping started")
    logger.info("=" * 60)

    raw_dirs = {
        "mgdb": PROJECT_ROOT / "data" / "raw" / "mgdb",
        "mgtbind": PROJECT_ROOT / "data" / "raw" / "mgtbind",
    }

    # Step 7: Extract unique protein names
    protein_names = extract_unique_proteins(raw_dirs)
    logger.info(f"Unique protein names found: {len(protein_names)}")

    if not protein_names:
        logger.error("No protein names extracted. Run download_data.py first.")
        sys.exit(1)

    names_df = pd.DataFrame({"original_name": protein_names})
    names_path = CURATED_DIR / "unique_protein_names.csv"
    names_df.to_csv(names_path, index=False)
    logger.info(f"Saved unique protein names: {names_path}")

    # Steps 8 & 9: UniProt queries
    all_candidate_rows = []
    n_ambiguous = 0
    n_no_match = 0
    n_api_error = 0

    for i, name in enumerate(protein_names, 1):
        logger.info(f"[{i}/{len(protein_names)}] Querying UniProt for: '{name}'")
        candidates = query_uniprot(name, max_results=MAX_CANDIDATES + 2)
        ambiguity = detect_ambiguity(candidates)

        if "AMBIGUOUS" in ambiguity:
            n_ambiguous += 1
        elif ambiguity == "NO_CONFIDENT_MATCH":
            n_no_match += 1
            logger.warning(f"  No confident match for: '{name}'")
        elif ambiguity == "API_ERROR":
            n_api_error += 1
            logger.error(f"  API error for: '{name}'")

        rows = build_candidate_rows(name, candidates, ambiguity)
        all_candidate_rows.extend(rows)
        time.sleep(UNIPROT_RATE_LIMIT_DELAY)

    candidates_df = pd.DataFrame(all_candidate_rows)
    cand_path = CURATED_DIR / "protein_mapping_candidates.csv"
    candidates_df.to_csv(cand_path, index=False)
    logger.info(f"Saved mapping candidates: {cand_path}")

    review_df = build_manual_review_table(protein_names, candidates_df)
    review_path = CURATED_DIR / "protein_mapping_manual_review.csv"
    review_df.to_csv(review_path, index=False)
    logger.info(f"Saved manual review table: {review_path}")

    # Summary
    logger.info("\n--- Protein Mapping Summary ---")
    logger.info(f"Total unique proteins : {len(protein_names)}")
    logger.info(f"Ambiguous mappings    : {n_ambiguous}")
    logger.info(f"No confident match    : {n_no_match}")
    logger.info(f"API errors            : {n_api_error}")
    logger.info(f"Ambiguity rate        : {n_ambiguous/len(protein_names)*100:.1f}%")
    logger.info("Phase 0 — Protein Mapping complete.")

    if not review_df["chosen_uniprot"].str.strip().any():
        logger.warning(
            "MANUAL REVIEW REQUIRED: protein_mapping_manual_review.csv "
            "must be completed by a human reviewer before Phase 1."
        )


if __name__ == "__main__":
    main()
