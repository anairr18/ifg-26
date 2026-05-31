"""
generate_hashes.py
------------------
Phase 0 — Step 4: Generate SHA-256 hashes for all raw files and create frozen snapshot.

Usage:
    python scripts/generate_hashes.py [--snapshot-dir DIRNAME]

Outputs:
    data/frozen_snapshot_YYYYMMDD/raw_hashes.txt
    Copies of all raw files in snapshot directory
"""

import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "mgdb",
    PROJECT_ROOT / "data" / "raw" / "mgtbind",
]

CHUNK_SIZE = 8192


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    snapshot_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    snapshot_dir = PROJECT_ROOT / "data" / f"frozen_snapshot_{snapshot_date}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    hash_file = snapshot_dir / "raw_hashes.txt"
    lines = []
    lines.append(f"# IFG-26 Phase 0 Raw File Snapshot")
    lines.append(f"# Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"# Algorithm: SHA-256")
    lines.append("")

    all_files = []
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            print(f"[WARNING] Raw directory not found: {raw_dir}", file=sys.stderr)
            continue
        dataset = raw_dir.name
        lines.append(f"## {dataset.upper()}")
        for fpath in sorted(raw_dir.iterdir()):
            if fpath.is_file():
                all_files.append((fpath, dataset))

    if not all_files:
        print("[ERROR] No raw files found. Run download_data.py first.", file=sys.stderr)
        sys.exit(1)

    failed = []
    for fpath, dataset in all_files:
        sha = sha256_file(fpath)
        size = fpath.stat().st_size
        line = f"{sha}  {fpath.name}  ({size:,} bytes)"
        print(f"Hashing: {fpath.name} -> {sha[:16]}...")
        lines.append(line)

        # Copy to snapshot
        dest = snapshot_dir / dataset / fpath.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fpath, dest)

        # Verify copy integrity
        copy_sha = sha256_file(dest)
        if copy_sha != sha:
            print(f"[ERROR] Snapshot integrity check FAILED for {fpath.name}", file=sys.stderr)
            failed.append(fpath.name)

    lines.append("")
    lines.append(f"# Total files: {len(all_files)}")
    lines.append(f"# Snapshot directory: {snapshot_dir}")

    hash_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nHash file written: {hash_file}")
    print(f"Snapshot directory: {snapshot_dir}")

    if failed:
        print(f"\n[ERROR] Integrity check failed for: {failed}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[OK] All {len(all_files)} files verified. Snapshot complete.")


if __name__ == "__main__":
    main()
