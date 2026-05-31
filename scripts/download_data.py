"""
download_data.py  (v2 - Phase 0 Rebuild)
-----------------------------------------
Phase 0 — Step 3: Download raw files for MGDB and MGTbind.

WHAT CHANGED FROM v1:
  - MGDB: endpoint is POST http://mgdb.idruglab.cn/download with JSON body
    {"fileName": "..."}, NOT static GET URLs (SPA catches all /download/* paths).
  - MGTbind: correct filenames are compounds.csv, complexes.csv, citations.csv
    (NOT interactions.csv / structures.csv — those return 404).
  - MGTbind: server SSL certificate is expired; using verify=False with explicit
    warning logged. This is documented as an insecure workaround; see
    docs/mgtbind_ssl_resolution.md.
  - HTML detection: abort immediately if response body starts with <!DOCTYPE html>.
  - Content-Type validation: reject text/html responses.
  - Minimum file size guard: reject files < 10 KB.
  - SHA256 computed immediately after each download.

Usage:
    python scripts/download_data.py [--dataset mgdb|mgtbind|all]
    python scripts/download_data.py --dry-run

Rules:
    - Do NOT rename original files.
    - Log every download with timestamp, filename, size, and SHA256.
    - Do NOT canonicalize or transform data.
    - Abort loudly on HTML response; do not silently write bad data.
"""

import argparse
import hashlib
import io
import json
import logging
import ssl
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "acquisition_log.txt"

# Minimum file size for a valid dataset file (10 KB).
MIN_VALID_SIZE_BYTES = 10_000

TIMEOUT_SECONDS = 120
CHUNK_SIZE = 8192
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# ---------------------------------------------------------------------------
# MGDB download spec
# Each tuple: (local_filename, fileName_param_for_POST_body)
# ---------------------------------------------------------------------------
MGDB_BASE = "http://mgdb.idruglab.cn"
MGDB_POST_ENDPOINT = "http://mgdb.idruglab.cn/download"
MGDB_FILES = [
    # (local filename,         POST body fileName param)
    ("mgdb_compounds.csv",     "MG Compound"),
    ("mgdb_bioactivity.csv",   "Activity Data"),
    ("mgdb_references.csv",    "Article Reference"),
    ("mgdb_patents.csv",       "Patent Reference"),
]

# ---------------------------------------------------------------------------
# MGTbind download spec (SSL verify=False — expired server cert workaround)
# ---------------------------------------------------------------------------
MGTBIND_BASE = "https://mgtbind.pkumdl.cn/static/download"
MGTBIND_FILES = [
    # (local filename,              remote filename)
    ("mgtbind_compounds.csv",   "compounds.csv"),   # 2.4 MB — confirmed real CSV
    ("mgtbind_complexes.csv",   "complexes.csv"),   # 5.2 MB — confirmed real CSV
    ("mgtbind_citations.csv",   "citations.csv"),   # 0.1 MB — confirmed real CSV
]

DATASETS_META = {
    "mgdb": {
        "portal":   "http://mgdb.idruglab.cn",
        "version":  "1.0 (released June 2025, published online Nov 2025)",
        "citation": (
            "Li C. et al. 'MGDB: a curated database for molecular glues.' "
            "Nucleic Acids Research. 2026;54(D1):D1488-D1499. DOI:10.1093/nar/gkaf1131"
        ),
    },
    "mgtbind": {
        "portal":   "https://mgtbind.pkumdl.cn",
        "version":  "2026 NAR database issue release; online ahead of print Oct 29, 2025",
        "citation": (
            "Zhu J. et al. 'MGTbind: a comprehensive database of molecular glue ternary "
            "interactome.' Nucleic Acids Research. 2025. DOI:10.1093/nar/gkaf1075"
        ),
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("download_data")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_response_content(first_chunk: bytes, url: str, content_type: str) -> None:
    """Raise ValueError if response looks like HTML (SPA error page) instead of CSV."""
    low = first_chunk[:128].lower()
    if b"<!doctype html" in low or b"<html" in low:
        raise ValueError(
            f"HTML DETECTED — server returned an HTML page instead of CSV data.\n"
            f"  URL          : {url}\n"
            f"  Content-Type : {content_type}\n"
            f"  First 128B   : {first_chunk[:128]!r}\n"
            f"This is likely a SPA catch-all returning index.html. "
            f"Fix the download URL or method and retry."
        )
    if "text/html" in content_type.lower():
        raise ValueError(
            f"CONTENT-TYPE REJECTION — server declared text/html.\n"
            f"  URL          : {url}\n"
            f"  Content-Type : {content_type}\n"
            f"Not a valid CSV response."
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# MGTbind: insecure SSL context (expired server certificate workaround)
# ---------------------------------------------------------------------------

def _make_insecure_ssl_context() -> ssl.SSLContext:
    """
    Returns an SSL context with certificate verification disabled.

    REASON: mgtbind.pkumdl.cn has an expired TLS certificate (confirmed
    2026-02-21). The server is reachable and returns valid data; the issue
    is purely the cert expiry. This workaround is documented in
    docs/mgtbind_ssl_resolution.md and is acceptable for private research
    download. Do NOT use in production or for secret data.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---------------------------------------------------------------------------
# Core download functions
# ---------------------------------------------------------------------------

def _write_file_with_progress(data: bytes, dest: Path, logger: logging.Logger) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)


def download_mgdb_file(
    post_endpoint: str,
    file_name_param: str,
    local_filename: str,
    dest_dir: Path,
    logger: logging.Logger,
    dry_run: bool = False,
) -> dict:
    """
    Download one MGDB file via POST to /download with {"fileName": <param>}.
    The SPA returns file content as binary response body (blob download).
    """
    dest = dest_dir / local_filename
    url_label = f"{post_endpoint} [POST fileName={file_name_param!r}]"

    if dry_run:
        print(f"[DRY RUN] POST {post_endpoint} {{fileName: {file_name_param!r}}} -> {dest}")
        return {"status": "DRY_RUN"}

    ts_start = datetime.now(timezone.utc).isoformat()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[Attempt {attempt}/{MAX_RETRIES}] POST {post_endpoint} "
                        f"{{fileName: {file_name_param!r}}}")
            payload = json.dumps({"fileName": file_name_param}).encode("utf-8")
            req = urllib.request.Request(
                post_endpoint,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                    "User-Agent": "Mozilla/5.0 (research download script)",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read()

            # Validate — reject HTML
            _validate_response_content(raw, url_label, content_type)

            if len(raw) < MIN_VALID_SIZE_BYTES:
                raise ValueError(
                    f"FILE TOO SMALL — {len(raw):,} bytes (minimum {MIN_VALID_SIZE_BYTES:,} B). "
                    f"Response may be an error message."
                )

            _write_file_with_progress(raw, dest, logger)
            sha = sha256_bytes(raw)
            ts_end = datetime.now(timezone.utc).isoformat()

            logger.info(f"SUCCESS | {local_filename} | {len(raw):,} bytes | SHA256={sha[:12]}...")
            return {
                "url": url_label,
                "dest": str(dest),
                "filename": local_filename,
                "file_size_bytes": len(raw),
                "sha256": sha,
                "download_start_utc": ts_start,
                "download_end_utc": ts_end,
                "status": "SUCCESS",
            }

        except (urllib.error.URLError, ValueError, OSError) as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

    logger.error(f"FAILED after {MAX_RETRIES} attempts: {url_label}")
    return {
        "url": url_label, "dest": str(dest), "filename": local_filename,
        "file_size_bytes": 0, "sha256": "N/A",
        "download_start_utc": ts_start,
        "download_end_utc": datetime.now(timezone.utc).isoformat(),
        "status": f"FAILED after {MAX_RETRIES} attempts",
    }


def download_mgtbind_file(
    remote_filename: str,
    local_filename: str,
    dest_dir: Path,
    logger: logging.Logger,
    ssl_ctx: ssl.SSLContext,
    dry_run: bool = False,
) -> dict:
    """
    Download one MGTbind CSV from /static/download/<remote_filename>.
    Uses an insecure SSL context because the server cert is expired.
    """
    url = f"{MGTBIND_BASE}/{remote_filename}"
    dest = dest_dir / local_filename

    if dry_run:
        print(f"[DRY RUN] GET {url} [SSL verify=False] -> {dest}")
        return {"status": "DRY_RUN"}

    ts_start = datetime.now(timezone.utc).isoformat()
    logger.warning(
        f"MGTbind SSL cert expired — downloading with verify=False. "
        f"See docs/mgtbind_ssl_resolution.md. File: {remote_filename}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[Attempt {attempt}/{MAX_RETRIES}] GET {url} (SSL verify=False)")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (research download script)", "Accept": "*/*"},
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS, context=ssl_ctx) as resp:
                content_type = resp.headers.get("Content-Type", "")
                content_length = int(resp.headers.get("Content-Length", 0))
                # Stream with progress
                raw_parts = []
                downloaded = 0
                if HAS_TQDM:
                    pbar = tqdm(total=content_length or None, unit="B",
                                unit_scale=True, desc=local_filename)
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    raw_parts.append(chunk)
                    downloaded += len(chunk)
                    if HAS_TQDM:
                        pbar.update(len(chunk))
                if HAS_TQDM:
                    pbar.close()
                raw = b"".join(raw_parts)

            # Validate
            _validate_response_content(raw, url, content_type)
            if len(raw) < MIN_VALID_SIZE_BYTES:
                raise ValueError(
                    f"FILE TOO SMALL — {len(raw):,} bytes (minimum {MIN_VALID_SIZE_BYTES:,} B)."
                )

            _write_file_with_progress(raw, dest, logger)
            sha = sha256_bytes(raw)
            ts_end = datetime.now(timezone.utc).isoformat()

            logger.info(f"SUCCESS | {local_filename} | {len(raw):,} bytes | SHA256={sha[:12]}...")
            return {
                "url": url,
                "dest": str(dest),
                "filename": local_filename,
                "file_size_bytes": len(raw),
                "sha256": sha,
                "download_start_utc": ts_start,
                "download_end_utc": ts_end,
                "status": "SUCCESS",
                "ssl_warning": "verify=False used — server cert expired",
            }

        except (urllib.error.URLError, ValueError, OSError) as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

    logger.error(f"FAILED after {MAX_RETRIES} attempts: {url}")
    return {
        "url": url, "dest": str(dest), "filename": local_filename,
        "file_size_bytes": 0, "sha256": "N/A",
        "download_start_utc": ts_start,
        "download_end_utc": datetime.now(timezone.utc).isoformat(),
        "status": f"FAILED after {MAX_RETRIES} attempts",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 0: Raw data download (v2 rebuild)")
    parser.add_argument(
        "--dataset",
        choices=["mgdb", "mgtbind", "all"],
        default="all",
        help="Which dataset(s) to download (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print download plan without executing",
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("Phase 0 — Data Acquisition started (v2 rebuild)")
    logger.info(f"Dataset target: {args.dataset}")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("DRY RUN MODE — no files will be downloaded")

    targets = ["mgdb", "mgtbind"] if args.dataset == "all" else [args.dataset]
    all_results = []
    ssl_ctx = _make_insecure_ssl_context()

    for ds_name in targets:
        meta = DATASETS_META[ds_name]
        logger.info(f"\n--- Dataset: {ds_name.upper()} ---")
        logger.info(f"Portal:   {meta['portal']}")
        logger.info(f"Version:  {meta['version']}")
        logger.info(f"Citation: {meta['citation']}")

        if ds_name == "mgdb":
            dest_dir = PROJECT_ROOT / "data" / "raw" / "mgdb"
            logger.info(f"Method:   POST {MGDB_POST_ENDPOINT} with JSON body {{fileName: <name>}}")
            for local_name, file_param in MGDB_FILES:
                result = download_mgdb_file(
                    MGDB_POST_ENDPOINT, file_param, local_name,
                    dest_dir, logger, dry_run=args.dry_run,
                )
                result["dataset"] = "mgdb"
                all_results.append(result)

        elif ds_name == "mgtbind":
            dest_dir = PROJECT_ROOT / "data" / "raw" / "mgtbind"
            logger.warning(
                "MGTbind: server TLS certificate expired. "
                "Downloading with SSL verification disabled (insecure workaround). "
                "See docs/mgtbind_ssl_resolution.md."
            )
            for local_name, remote_name in MGTBIND_FILES:
                result = download_mgtbind_file(
                    remote_name, local_name,
                    dest_dir, logger, ssl_ctx,
                    dry_run=args.dry_run,
                )
                result["dataset"] = "mgtbind"
                all_results.append(result)

    if not args.dry_run:
        logger.info("\n--- Download Summary ---")
        ok   = [r for r in all_results if r.get("status") == "SUCCESS"]
        fail = [r for r in all_results if r.get("status", "").startswith("FAILED")]
        logger.info(f"Successful downloads : {len(ok)}")
        logger.info(f"Failed downloads     : {len(fail)}")
        if fail:
            logger.error("FAILED FILES:")
            for f in fail:
                logger.error(f"  {f['url']} -> {f['status']}")
            sys.exit(1)
        else:
            logger.info("All downloads successful. Ready for generate_hashes.py.")

    logger.info("Phase 0 — Data Acquisition complete.")


if __name__ == "__main__":
    main()
