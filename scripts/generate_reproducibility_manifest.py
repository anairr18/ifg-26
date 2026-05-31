"""
generate_reproducibility_manifest.py
--------------------------------------
Generates docs/reproducibility_manifest.md with file hashes,
pipeline config, and environment details.
"""
import hashlib, json, logging, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent

def sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "MISSING"

def get_rdkit_version():
    try:
        from rdkit import __version__
        return __version__
    except Exception:
        return "unknown"

def main():
    lg = logging.getLogger("manifest")
    lg.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    lg.addHandler(sh)
    ts = datetime.now(timezone.utc).isoformat()

    key_files = {
        "ChEMBL universe": ROOT / "data/external_chembl_universe.parquet",
        "External PMD v2": ROOT / "data/phase5_external_pmd_v2.parquet",
        "Core ladder results": ROOT / "data/phase5_negative_ladder_core.csv",
        "Scaffold audit": ROOT / "data/external_pmd_scaffold_audit.csv",
        "Positive pool (scaffold)": ROOT / "data/pu/pool_P_scaffold.parquet",
        "Unlabeled pool (scaffold)": ROOT / "data/pu/pool_U_scaffold.parquet",
        "Canonicalized compounds": ROOT / "dataset/phase1/canonicalized_compounds.csv",
        "Test scaffold split": ROOT / "dataset/phase2/test_scaffold.csv",
    }

    hash_table = ""
    for label, path in key_files.items():
        h = sha256(path)
        size = f"{path.stat().st_size / 1024 / 1024:.1f} MB" if path.exists() else "MISSING"
        hash_table += f"| {label} | {size} | `{h[:16]}...` |\n"

    manifest = f"""# IFG-26 Reproducibility Manifest

_Generated: {ts}_

## Environment

| Component | Version |
|---|---|
| Python | 3.10+ |
| RDKit | {get_rdkit_version()} |
| ChEMBL Dataset | ChEMBL 34 (`chembl_34_chemreps.txt.gz`) |
| Conda Environment | `ifg26` |

## Pipeline Configuration

| Parameter | Value |
|---|---|
| ECFP radius | 2 |
| ECFP bit size | 2048 |
| Tanimoto window (PMD v2) | 0.35 – 0.65 |
| MW tolerance (PMD v2) | ±7% |
| LogP tolerance (PMD v2) | ±0.7 |
| Lipinski MW cutoff | < 500 |
| Lipinski LogP cutoff | < 5 |
| Lipinski HBD cutoff | < 5 |
| Lipinski HBA cutoff | < 10 |
| Random seed | 42 |

## Dataset Hashes (SHA-256, first 16 chars)

| File | Size | SHA-256 (prefix) |
|---|---|---|
{hash_table}

## Random Seeds

All random operations use `random_state=42` or `np.random.seed(42)`.

## Pipeline Execution Order

```bash
# Step 1: Download ChEMBL universe
python scripts/phase5A_download_external_universe.py

# Step 2: Generate External PMD v2 (documented failure)
python scripts/phase5B_generate_external_pmd_v2.py

# Step 3: Core Negative Realism Ladder
python scripts/phase5B_negative_ladder.py

# Step 4: Forensic Separability Audit
python scripts/phase5E_forensic_separability_analysis.py

# Step 5: External PMD Failure Analysis
python scripts/phase5F_external_pmd_failure_analysis.py

# Step 6: Compile Final Report
python scripts/phase5_compile_report.py
```

## ChEMBL Citation

> Zdrazil B, et al. (2024). The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity complementary databases. Nucleic Acids Research.
> https://doi.org/10.1093/nar/gkad1004
"""
    out = ROOT / "docs/reproducibility_manifest.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(manifest)
    lg.info(f"Reproducibility manifest saved to {out}")

if __name__ == "__main__":
    main()
