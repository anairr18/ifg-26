"""
raw_schema_inspection.py
------------------------
Phase 0 — Step 5: Raw schema inspection of MGDB and MGTbind datasets.

Usage:
    python scripts/raw_schema_inspection.py

Outputs:
    dataset/raw_schema_summary.csv
    docs/raw_data_audit_report.md (appended)
    Console summary
"""

import json
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

# Critical columns that MUST exist per dataset
CRITICAL_COLUMNS = {
    "mgdb": {
        "smiles": ["smiles", "canonical_smiles", "SMILES", "structure"],
        "protein": ["e3_ligase", "neo_substrate", "target", "protein"],
        "reference": ["reference", "ref", "pmid", "doi", "citation"],
        "endpoint": ["activity_type", "endpoint", "assay_type", "activity"],
    },
    "mgtbind": {
        "smiles": ["smiles", "canonical_smiles", "SMILES", "mol_smiles"],
        "protein": ["e3_ligase", "neo_substrate", "protein_1", "protein_2", "target"],
        "reference": ["reference", "source", "pmid", "doi"],
        "endpoint": ["activity_type", "endpoint", "bioactivity_type", "measurement_type"],
    },
}

PROTEIN_LIKE_COLS = [
    "e3_ligase", "neo_substrate", "target", "protein", "protein_1",
    "protein_2", "receptor", "ligase",
]
ASSAY_LIKE_COLS = [
    "activity_type", "assay_type", "endpoint", "bioactivity_type",
    "measurement_type", "assay",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first matching column name (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def inspect_file(fpath: Path, dataset_name: str) -> dict:
    """Inspect a single CSV file and return a stats dict."""
    print(f"\nInspecting: {fpath.name}")

    try:
        df = pd.read_csv(fpath, low_memory=False)
    except Exception as e:
        return {"file": fpath.name, "error": str(e)}

    n_rows, n_cols = df.shape
    stats = {
        "file": fpath.name,
        "dataset": dataset_name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "missing_pct": {},
        "unique_proteins": [],
        "unique_assay_types": [],
        "unique_endpoint_types": [],
        "critical_column_flags": {},
        "error": None,
    }

    # Missingness per column
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        stats["missing_pct"][col] = round(pct, 2)

    # Critical column presence check
    crit = CRITICAL_COLUMNS.get(dataset_name, {})
    for role, candidates in crit.items():
        found = find_col(df, candidates)
        stats["critical_column_flags"][role] = {
            "found": found is not None,
            "matched_col": found,
        }
        if found is None:
            print(f"  [WARNING] Critical column '{role}' NOT FOUND (tried: {candidates})")

    # Unique protein values
    for proto_col in PROTEIN_LIKE_COLS:
        actual = find_col(df, [proto_col])
        if actual:
            vals = df[actual].dropna().astype(str).unique().tolist()
            stats["unique_proteins"].extend(vals)
    stats["unique_proteins"] = sorted(set(stats["unique_proteins"]))

    # Unique assay types
    for assay_col in ASSAY_LIKE_COLS:
        actual = find_col(df, [assay_col])
        if actual:
            vals = df[actual].dropna().astype(str).unique().tolist()
            stats["unique_assay_types"].extend(vals)
    stats["unique_assay_types"] = sorted(set(stats["unique_assay_types"]))

    # Unique endpoint types (same candidates, different interpretation)
    stats["unique_endpoint_types"] = stats["unique_assay_types"]  # refined in endpoint_audit.py

    print(f"  Rows: {n_rows:,} | Cols: {n_cols}")
    print(f"  Columns: {list(df.columns)}")
    critical_ok = all(v["found"] for v in stats["critical_column_flags"].values())
    print(f"  Critical columns present: {'YES' if critical_ok else 'NO — SEE FLAGS ABOVE'}")

    return stats


def build_summary_df(all_stats: list[dict]) -> pd.DataFrame:
    rows = []
    for s in all_stats:
        if s.get("error"):
            rows.append({"file": s["file"], "dataset": s.get("dataset", "?"),
                         "n_rows": "ERROR", "n_cols": "ERROR", "note": s["error"]})
            continue
        max_miss_col = max(s["missing_pct"], key=s["missing_pct"].get) if s["missing_pct"] else "N/A"
        max_miss_val = max(s["missing_pct"].values()) if s["missing_pct"] else 0
        rows.append({
            "file": s["file"],
            "dataset": s["dataset"],
            "n_rows": s["n_rows"],
            "n_cols": s["n_cols"],
            "max_missing_col": max_miss_col,
            "max_missing_pct": max_miss_val,
            "smiles_col_found": s["critical_column_flags"].get("smiles", {}).get("found", False),
            "protein_col_found": s["critical_column_flags"].get("protein", {}).get("found", False),
            "reference_col_found": s["critical_column_flags"].get("reference", {}).get("found", False),
            "endpoint_col_found": s["critical_column_flags"].get("endpoint", {}).get("found", False),
            "n_unique_proteins": len(s["unique_proteins"]),
            "n_unique_assays": len(s["unique_assay_types"]),
        })
    return pd.DataFrame(rows)


def write_audit_section(all_stats: list[dict], df_summary: pd.DataFrame) -> None:
    """Append schema inspection section to raw_data_audit_report.md."""
    audit_path = DOCS_DIR / "raw_data_audit_report.md"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        "",
        "---",
        "",
        "## Section 2 — Raw Schema Inspection",
        f"_Generated: {ts}_",
        "",
        "### Summary Table",
        "",
        df_summary.to_markdown(index=False),
        "",
        "### Per-File Details",
        "",
    ]

    for s in all_stats:
        if s.get("error"):
            lines.append(f"#### {s['file']}\n**ERROR:** {s['error']}\n")
            continue
        lines.append(f"#### {s['file']} ({s['dataset'].upper()})")
        lines.append(f"- Rows: {s['n_rows']:,}")
        lines.append(f"- Columns ({s['n_cols']}): `{', '.join(s['columns'])}`")
        lines.append(f"- Critical columns: {json.dumps(s['critical_column_flags'], indent=2)}")
        lines.append(f"- Unique proteins ({len(s['unique_proteins'])}): {s['unique_proteins'][:20]}")
        lines.append(f"- Unique assay types: {s['unique_assay_types'][:20]}")
        miss_top = sorted(s["missing_pct"].items(), key=lambda x: -x[1])[:5]
        lines.append(f"- Top missing columns: {miss_top}")
        lines.append("")

    with open(audit_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nAudit report updated: {audit_path}")


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

    all_stats = []
    for ds_name, raw_dir in raw_dirs.items():
        if not raw_dir.exists():
            print(f"[WARNING] Missing raw dir: {raw_dir}", file=sys.stderr)
            continue
        for fpath in sorted(raw_dir.glob("*.csv")):
            s = inspect_file(fpath, ds_name)
            all_stats.append(s)

    if not all_stats:
        print("[ERROR] No CSV files found in raw directories. Run download_data.py first.", file=sys.stderr)
        sys.exit(1)

    df_summary = build_summary_df(all_stats)

    out_csv = CURATED_DIR / "raw_schema_summary.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSchema summary saved: {out_csv}")

    write_audit_section(all_stats, df_summary)

    # Save full stats as JSON for downstream scripts
    import json
    json_out = CURATED_DIR / "raw_schema_full.json"
    # Convert lists to JSON-serializable format
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"Full stats JSON saved: {json_out}")

    # Check for critical failures
    critical_failures = []
    for s in all_stats:
        if s.get("error"):
            continue
        for role, info in s.get("critical_column_flags", {}).items():
            if not info["found"]:
                critical_failures.append(f"{s['file']} missing '{role}' column")

    if critical_failures:
        print("\n[CRITICAL] Missing critical columns:")
        for f in critical_failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\n[OK] All critical columns found.")


if __name__ == "__main__":
    main()
