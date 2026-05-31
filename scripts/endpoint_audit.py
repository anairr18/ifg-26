"""
endpoint_audit.py
-----------------
Phase 0 — Step 10: Extract and classify unique endpoint types into taxonomy.

Usage:
    python scripts/endpoint_audit.py

Outputs:
    docs/task_definition.md
    dataset/endpoint_taxonomy.csv
    Appends to docs/raw_data_audit_report.md
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
DOCS_DIR = PROJECT_ROOT / "docs"

ENDPOINT_COL_CANDIDATES = [
    "activity_type", "assay_type", "endpoint", "bioactivity_type",
    "measurement_type", "assay", "measurement",
]

# Taxonomy classification rules — keyword-based priority matching.
# Each entry: (category, [keywords to search for in lowercase endpoint string])
TAXONOMY_RULES: list[tuple[str, list[str]]] = [
    ("degradation_evidence",   ["degradation", "degron", "dc50", "dmax", "proteasome", "ubiquitin", "ub", "deg"]),
    ("binding_evidence",       ["binding", "ic50", "ki", "kd", "affinity", "spr", "itc", "tr-fret", "htrf"]),
    ("cooperativity_evidence", ["cooperativity", "alpha", "ternary", "cooperative", "proximity"]),
    ("cellular_phenotype",     ["cell", "viability", "proliferation", "apoptosis", "growth", "phenotype", "antiproliferative"]),
]


def classify_endpoint(endpoint_str: str) -> str:
    if not isinstance(endpoint_str, str):
        return "other"
    lower = endpoint_str.lower()
    for category, keywords in TAXONOMY_RULES:
        if any(kw in lower for kw in keywords):
            return category
    return "other"


def find_endpoint_cols(df: pd.DataFrame) -> list[str]:
    lower_map = {c.lower(): c for c in df.columns}
    found = []
    for cand in ENDPOINT_COL_CANDIDATES:
        if cand.lower() in lower_map and lower_map[cand.lower()] not in found:
            found.append(lower_map[cand.lower()])
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    raw_dirs = {
        "mgdb": PROJECT_ROOT / "data" / "raw" / "mgdb",
        "mgtbind": PROJECT_ROOT / "data" / "raw" / "mgtbind",
    }

    all_endpoints = []

    for ds_name, raw_dir in raw_dirs.items():
        if not raw_dir.exists():
            continue
        for fpath in sorted(raw_dir.glob("*.csv")):
            try:
                df = pd.read_csv(fpath, low_memory=False)
            except Exception:
                continue
            cols = find_endpoint_cols(df)
            for col in cols:
                vals = df[col].dropna().astype(str).unique()
                for v in vals:
                    v = v.strip()
                    if v and v.lower() != "nan":
                        all_endpoints.append({
                            "source_file": fpath.name,
                            "dataset": ds_name,
                            "source_column": col,
                            "raw_endpoint": v,
                            "taxonomy_category": classify_endpoint(v),
                        })

    if not all_endpoints:
        print("[WARNING] No endpoint data found. Download data first.", file=sys.stderr)
        # Still generate template documents
        endpoint_df = pd.DataFrame(columns=["source_file", "dataset", "source_column",
                                             "raw_endpoint", "taxonomy_category"])
    else:
        endpoint_df = pd.DataFrame(all_endpoints)

    # Save taxonomy CSV
    taxonomy_path = CURATED_DIR / "endpoint_taxonomy.csv"
    endpoint_df.to_csv(taxonomy_path, index=False)
    print(f"Endpoint taxonomy saved: {taxonomy_path}")

    # Distribution summary
    if not endpoint_df.empty:
        dist = endpoint_df["taxonomy_category"].value_counts().to_dict()
    else:
        dist = {cat: 0 for cat, _ in TAXONOMY_RULES}
        dist["other"] = 0

    # Write task_definition.md
    ts = datetime.now(timezone.utc).isoformat()
    task_def_path = DOCS_DIR / "task_definition.md"
    task_def = f"""# IFG-26 Task Definition

_Generated: {ts}_
_Status: **REQUIRES HUMAN REVIEW BEFORE PHASE 1**_

---

## 1. Endpoint Taxonomy

The following endpoint types were extracted from MGDB and MGTbind raw data
and mapped to a four-category taxonomy for benchmark task definition.

| Category | Description | Count (raw) |
|---|---|---|
| `degradation_evidence` | DC50, Dmax, ubiquitination assays, proteasome-dependent activity | {dist.get('degradation_evidence', 0)} |
| `binding_evidence` | IC50, Kd, Ki, SPR, HTRF, TR-FRET binary/ternary binding assays | {dist.get('binding_evidence', 0)} |
| `cooperativity_evidence` | Alpha cooperativity, ternary complex formation, proximity-based assays | {dist.get('cooperativity_evidence', 0)} |
| `cellular_phenotype` | Cell viability, antiproliferative activity, apoptosis, phenotypic assays | {dist.get('cellular_phenotype', 0)} |
| `other` | Unclassified or ambiguous endpoint types | {dist.get('other', 0)} |

---

## 2. Proposed Primary Benchmark Task

**Task: Binary Classification — Molecular Glue Activity**

- **Input:** Molecular structure (SMILES) + E3 ligase + Neo-substrate (protein pair)
- **Label:** Active (has degradation or binding evidence) vs Inactive
- **Rationale:** Most datasets include IC50/DC50 measurements enabling binary thresholding.
  This task maximizes data volume across both MGDB and MGTbind.

---

## 3. Alternative Benchmark Tasks

| Task | Description | Feasibility |
|---|---|---|
| Regression (DC50) | Predict DC50 values for degradation | Limited — subset with quantitative Dmax/DC50 |
| Ternary affinity prediction | Predict cooperative binding affinity (alpha) | Emerging — MGTbind ternary data |
| E3 ligase specificity | Classify E3 ligase engaged by compound | Medium — sufficient label diversity |
| Cooperativity classification | Binary: cooperative vs non-cooperative | Limited — few cooperativity measurements |

---

## 4. Justification

1. **Coverage:** Binary classification on combined MGDB+MGTbind maximizes training signal.
2. **Literature precedent:** Existing molecular glue ML papers (e.g., MEGAcell, MolGlue)
   mainly approach binary classification tasks.
3. **Data quality:** DC50 regression requires high-quality quantitative data — feasibility
   should be re-evaluated after Phase 1 data cleaning.
4. **Benchmark integrity:** Ternary affinity prediction is scientifically valuable but
   currently data-limited; recommend as a secondary benchmark.

---

## 5. Pre-Phase 1 Human Review Items

> ⚠️ The following items MUST be resolved by a domain expert before Phase 1:
>
> - [ ] Confirm primary task label definition and threshold
> - [ ] Validate endpoint taxonomy mappings above
> - [ ] Decide whether to combine MGDB + MGTbind or benchmark separately
> - [ ] Confirm handling of multi-endpoint compounds
> - [ ] Confirm handling of incomplete ternary measurements

---

_This document was auto-generated. All decisions above require domain expert validation._
"""
    task_def_path.write_text(task_def, encoding="utf-8")
    print(f"Task definition written: {task_def_path}")

    # Append to audit report
    audit_path = DOCS_DIR / "raw_data_audit_report.md"
    section = f"""
---

## Section 4 — Endpoint Taxonomy

_Generated: {ts}_

### Distribution

| Category | Count |
|---|---|
"""
    for cat, count in dist.items():
        section += f"| `{cat}` | {count} |\n"
    section += "\n_See `dataset/endpoint_taxonomy.csv` for full per-endpoint mapping._\n"

    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"Audit report updated: {audit_path}")
    print("[OK] Endpoint audit complete.")


if __name__ == "__main__":
    main()
