"""
stereo_audit.py
---------------
Phase 0 — Step 6: SMILES integrity and stereochemistry audit.

Usage:
    python scripts/stereo_audit.py

Outputs:
    dataset/smiles_integrity_report.csv
    docs/raw_data_audit_report.md (appended)

Rules:
    - Do NOT canonicalize or fix SMILES.
    - Report only; do not modify raw data.
    - Uses RDKit for validation.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    print("[WARNING] RDKit not available. SMILES validation will be limited.", file=sys.stderr)
    RDKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
DOCS_DIR = PROJECT_ROOT / "docs"

# Potential SMILES column names (case-insensitive matching)
SMILES_COL_CANDIDATES = ["smiles", "canonical_smiles", "SMILES", "structure", "mol_smiles"]

# Salt indicators: common counterions as fragments
KNOWN_SALT_FRAGMENTS = [
    "[Na+]", "[K+]", "[Ca+2]", "[Mg+2]", "[Cl-]", "[Br-]",
    "[NH4+]", "[Li+]", "CC(=O)[O-]",  # acetate
]

GO_NO_GO_INVALID_THRESHOLD = 0.10   # >10% invalid SMILES → flag
GO_NO_GO_STEREO_THRESHOLD = 0.30    # >30% missing stereo → document limitation


# ---------------------------------------------------------------------------
# SMILES classification
# ---------------------------------------------------------------------------

def is_valid_smiles(smi: str) -> bool:
    """Returns True if RDKit can parse the SMILES."""
    if not RDKIT_AVAILABLE:
        return True  # cannot validate without RDKit
    if not isinstance(smi, str) or not smi.strip():
        return False
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    return mol is not None


def is_multifragment(smi: str) -> bool:
    """Returns True if SMILES contains '.' (multiple components)."""
    return isinstance(smi, str) and "." in smi


def looks_like_salt(smi: str) -> bool:
    """Heuristic: True if any known salt fragment string appears in SMILES."""
    if not isinstance(smi, str):
        return False
    return any(frag in smi for frag in KNOWN_SALT_FRAGMENTS) or "." in smi


def missing_stereo(smi: str) -> bool:
    """Returns True if SMILES lacks '@' and '/' markers entirely."""
    if not isinstance(smi, str):
        return True
    return "@" not in smi and "/" not in smi


def aromatic_inconsistency(smi: str) -> bool:
    """
    Very rough heuristic: flags if SMILES uses lowercase (aromatic) atoms
    mixed with uppercase in unusual patterns.
    RDKit does proper validation; here we just flag any parse error.
    """
    if not RDKIT_AVAILABLE or not isinstance(smi, str):
        return False
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return False
    except Exception:
        return True


def audit_smiles_column(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Audit a SMILES column and return a results DataFrame."""
    results = []
    for idx, row in df.iterrows():
        smi = row[smiles_col]
        results.append({
            "row_idx": idx,
            "smiles": smi,
            "is_valid": is_valid_smiles(smi) if isinstance(smi, str) and smi.strip() else False,
            "is_empty": not isinstance(smi, str) or not smi.strip(),
            "is_multifragment": is_multifragment(smi),
            "looks_like_salt": looks_like_salt(smi),
            "missing_stereo": missing_stereo(smi),
            "aromatic_inconsistency": aromatic_inconsistency(smi),
        })
    return pd.DataFrame(results)


def find_smiles_col(df: pd.DataFrame) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in SMILES_COL_CANDIDATES:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_stats(audit_df: pd.DataFrame, n_total: int) -> dict:
    n = len(audit_df)
    if n == 0:
        return {}
    return {
        "n_total_rows": n_total,
        "n_smiles_entries": n,
        "pct_empty": round(audit_df["is_empty"].mean() * 100, 2),
        "pct_invalid": round((~audit_df["is_valid"] & ~audit_df["is_empty"]).mean() * 100, 2),
        "pct_multifragment": round(audit_df["is_multifragment"].mean() * 100, 2),
        "pct_salt": round(audit_df["looks_like_salt"].mean() * 100, 2),
        "pct_missing_stereo": round(audit_df["missing_stereo"].mean() * 100, 2),
        "pct_aromatic_inconsistency": round(audit_df["aromatic_inconsistency"].mean() * 100, 2),
    }


def go_no_go(stats: dict, file_name: str) -> list[str]:
    flags = []
    if stats.get("pct_invalid", 0) > GO_NO_GO_INVALID_THRESHOLD * 100:
        flags.append(
            f"[MAJOR INTEGRITY ISSUE] {file_name}: "
            f"{stats['pct_invalid']:.1f}% invalid SMILES "
            f"(threshold: {GO_NO_GO_INVALID_THRESHOLD * 100:.0f}%)"
        )
    if stats.get("pct_missing_stereo", 0) > GO_NO_GO_STEREO_THRESHOLD * 100:
        flags.append(
            f"[LIMITATION] {file_name}: "
            f"{stats['pct_missing_stereo']:.1f}% SMILES missing stereochemistry markers. "
            "Document as limitation for stereoanalysis."
        )
    return flags


def write_audit_section(summary_rows: list[dict], go_no_go_flags: list[str]) -> None:
    audit_path = DOCS_DIR / "raw_data_audit_report.md"
    ts = datetime.now(timezone.utc).isoformat()
    df = pd.DataFrame(summary_rows)

    lines = [
        "",
        "---",
        "",
        "## Section 3 — SMILES Integrity Audit",
        f"_Generated: {ts}_",
        f"_RDKit available: {RDKIT_AVAILABLE}_",
        "",
        "### Summary",
        "",
        df.to_markdown(index=False) if not df.empty else "_No data_",
        "",
        "### Go/No-Go Flags",
        "",
    ]
    if go_no_go_flags:
        for flag in go_no_go_flags:
            lines.append(f"- ⚠️  {flag}")
    else:
        lines.append("- ✅ All SMILES integrity checks passed.")
    lines.append("")

    with open(audit_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Audit report updated: {audit_path}")


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

    all_audit_rows = []
    summary_rows = []
    go_no_go_flags = []

    for ds_name, raw_dir in raw_dirs.items():
        if not raw_dir.exists():
            print(f"[WARNING] Missing raw dir: {raw_dir}", file=sys.stderr)
            continue
        for fpath in sorted(raw_dir.glob("*.csv")):
            print(f"\nAuditing SMILES in: {fpath.name}")
            try:
                df = pd.read_csv(fpath, low_memory=False)
            except Exception as e:
                print(f"  [ERROR] Cannot read file: {e}", file=sys.stderr)
                continue

            smiles_col = find_smiles_col(df)
            if not smiles_col:
                print(f"  [WARNING] No SMILES column found in {fpath.name}. Skipping.")
                continue

            audit_df = audit_smiles_column(df, smiles_col)
            audit_df["file"] = fpath.name
            audit_df["dataset"] = ds_name
            all_audit_rows.append(audit_df)

            stats = compute_stats(audit_df, len(df))
            stats["file"] = fpath.name
            stats["dataset"] = ds_name
            stats["smiles_col"] = smiles_col
            summary_rows.append(stats)

            flags = go_no_go(stats, fpath.name)
            go_no_go_flags.extend(flags)

            print(f"  Valid SMILES   : {100 - stats['pct_invalid']:.1f}%")
            print(f"  Multi-fragment : {stats['pct_multifragment']:.1f}%")
            print(f"  Salt-like      : {stats['pct_salt']:.1f}%")
            print(f"  Missing stereo : {stats['pct_missing_stereo']:.1f}%")
            if flags:
                for f in flags:
                    print(f"  *** {f}")

    if not all_audit_rows:
        print("[ERROR] No SMILES data to audit. Run download_data.py first.", file=sys.stderr)
        sys.exit(1)

    # Save row-level audit report
    full_audit = pd.concat(all_audit_rows, ignore_index=True)
    out_path = CURATED_DIR / "smiles_integrity_report.csv"
    full_audit.to_csv(out_path, index=False)
    print(f"\nSMILES integrity report saved: {out_path}")

    write_audit_section(summary_rows, go_no_go_flags)

    if any("MAJOR INTEGRITY ISSUE" in f for f in go_no_go_flags):
        print("\n[CHECK REQUIRED] Major SMILES integrity issues detected. Review before Phase 1.")
        sys.exit(2)  # exit code 2 = non-fatal flag requiring review
    else:
        print("\n[OK] SMILES integrity audit complete.")


if __name__ == "__main__":
    main()
