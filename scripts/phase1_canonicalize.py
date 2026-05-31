"""
phase1_canonicalize.py
======================
IFG-26 Phase 1 — Fast Chemical Standardization.

Config: configs/experiment/phase1_default.yaml
Usage:  python scripts/phase1_canonicalize.py [--config <path>]

Outputs (in dataset/phase1/):
    mgdb_compounds_canonicalized.csv
    mgtbind_compounds_canonicalized.csv
    chem_failures.csv
    phase1_chem_audit.json
    logs/phase1_chem.log

Design decisions (see also config YAML):
  - tautomer_mode: off  (Phase 1 default)
      MGDB contains many large macrocyclic PROTACs / bifunctional degraders.
      TautomerEnumerator hits 300-900 tautomers on them → seconds per molecule.
      Disabled for Phase 1. InChIKey via standardized InChI still handles
      tautomer-invariant dedup. Re-enable in Phase 1b with caps.

  - No stereo imputation. stereo_status in {defined, undefined, none}.

  - Structured failure logging: every failed molecule is written to
    chem_failures.csv with a failure_reason enum.

  - Performance ETA printed every log_every_n_rows rows.
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

# Suppress RDKit stderr noise — failures are captured explicitly.
warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError as e:
    print(f"[FATAL] RDKit not available: {e}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase1_default.yaml"

# ---------------------------------------------------------------------------
# Failure reasons (enum-like strings for chem_failures.csv)
# ---------------------------------------------------------------------------
FAIL_EMPTY             = "empty_smiles"
FAIL_PROTEIN_SEQ       = "protein_sequence_contamination"
FAIL_MARKUSH           = "markush_star"
FAIL_PARSE             = "parse_fail"
FAIL_SANITIZE          = "sanitize_fail"
FAIL_INCHI             = "inchi_fail"
FAIL_SALT_STRIP        = "salt_strip_fail"

# Pattern for protein sequence contamination: mostly uppercase letters,
# no ring closures or bond chars, starts with common amino acid initials.
_AA_CHARS = set("ACDEFGHIKLMNPQRSTVWY")
_SMILES_SPECIFIC = set("()[]@#=/.\\+-%0123456789")

def _looks_like_protein_seq(s: str) -> bool:
    chars = set(s.upper())
    return (
        len(s) > 20
        and chars.issubset(_AA_CHARS | {" ", "\n"})
        and not chars.intersection(_SMILES_SPECIFIC)
    )

# ---------------------------------------------------------------------------
# Standardization objects (module-level, created once)
# ---------------------------------------------------------------------------
_lfc      = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def _stereo_status(mol: Chem.Mol) -> str:
    """Classify stereo: defined / undefined / none. Never imputes."""
    try:
        si = Chem.FindPotentialStereo(mol)
        has_defined   = any(s.specified == Chem.StereoSpecified.Specified for s in si)
        has_undefined = any(s.specified != Chem.StereoSpecified.Specified for s in si)
        if has_defined:
            return "defined"
        if has_undefined:
            return "undefined"
    except Exception:
        pass
    return "none"


def canonicalize_smiles(
    smiles: str,
    source_file: str,
    row_id: str,
    cfg: dict,
    failure_rows: list,
) -> dict | None:
    """
    Canonicalize one SMILES string.

    Returns a dict of output columns on success, None on failure.
    Appends failure records to failure_rows list in-place.

    Args:
        smiles:        raw SMILES string from source file
        source_file:   name of source CSV (for failure log)
        row_id:        compound ID / row index (for failure log)
        cfg:           canonicalization config dict
        failure_rows:  accumulator for chem_failures.csv rows
    """
    raw = smiles

    def _fail(reason: str, exc: str = "") -> None:
        failure_rows.append({
            "source_file":   source_file,
            "row_id":        row_id,
            "raw_smiles":    str(raw or "")[:300],
            "failure_reason": reason,
            "exception_str": str(exc)[:200],
        })

    # ── Guard: empty ──────────────────────────────────────────────────
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        _fail(FAIL_EMPTY)
        return None

    smiles = smiles.strip()

    # ── Guard: protein sequence contamination ─────────────────────────
    if _looks_like_protein_seq(smiles):
        _fail(FAIL_PROTEIN_SEQ)
        return None

    # ── Guard: Markush / wildcard ─────────────────────────────────────
    if "*" in smiles or "[*]" in smiles:
        _fail(FAIL_MARKUSH)
        return None

    # ── Parse ─────────────────────────────────────────────────────────
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        _fail(FAIL_PARSE, str(e))
        return None

    if mol is None:
        _fail(FAIL_PARSE)
        return None

    # ── Sanitize (should already be done by MolFromSmiles; belt+braces) ──
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        _fail(FAIL_SANITIZE, str(e))
        return None

    # ── Largest fragment (salt strip) ─────────────────────────────────
    if cfg.get("largest_fragment", True):
        try:
            mol = _lfc.choose(mol)
        except Exception as e:
            _fail(FAIL_SALT_STRIP, str(e))
            return None
        if mol is None:
            _fail(FAIL_SALT_STRIP)
            return None

    # ── Uncharge / formal charge normalisation ────────────────────────
    if cfg.get("uncharge", True):
        try:
            mol = _uncharger.uncharge(mol)
        except Exception:
            pass  # Non-fatal; proceed with current mol

    # ── NOTE: tautomer mode ───────────────────────────────────────────
    # tautomer_mode is checked here for forward-compatibility.
    # Phase 1 default: off.
    # Phase 1b (future): instantiate TautomerEnumerator with caps from config.
    tautomer_mode = cfg.get("tautomer_mode", "off")
    if tautomer_mode == "on":
        caps = cfg.get("tautomer_caps", {})
        # Phase 1b implementation note:
        #   te = rdMolStandardize.TautomerEnumerator()
        #   te.SetMaxTautomers(caps.get("max_tautomers", 50))
        #   te.SetMaxTransforms(caps.get("max_transforms", 200))
        #   mol = te.Canonicalize(mol)  # with per-molecule timeout
        # Not implemented: raise error to prevent silent skip.
        raise NotImplementedError(
            "tautomer_mode=on is reserved for Phase 1b. "
            "Implement per-molecule timeout before enabling."
        )
    # tautomer_mode == "off": fall through, no tautomer step.

    # ── Canonical SMILES ──────────────────────────────────────────────
    try:
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        canonical_isomeric = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception as e:
        _fail(FAIL_SANITIZE, f"MolToSmiles: {e}")
        return None

    # ── InChI + InChIKey ─────────────────────────────────────────────
    try:
        inchi = MolToInchi(mol)
        if not inchi:
            _fail(FAIL_INCHI)
            return None
        inchi_key = InchiToInchiKey(inchi)
        if not inchi_key:
            _fail(FAIL_INCHI, "InchiToInchiKey returned None")
            return None
    except Exception as e:
        _fail(FAIL_INCHI, str(e))
        return None

    # ── Stereo status (no imputation) ────────────────────────────────
    stereo = _stereo_status(mol)

    # ── Molecular weight (for audit / sanity) ────────────────────────
    try:
        mw = rdMolDescriptors.CalcExactMolWt(mol)
    except Exception:
        mw = None

    return {
        "canonical_smiles":         canonical_smiles,
        "canonical_isomeric_smiles": canonical_isomeric,
        "inchi":                    inchi,
        "inchi_key":                inchi_key,
        "stereo_status":            stereo,
        "mol_weight":               round(mw, 4) if mw is not None else None,
        "tautomer_mode":            tautomer_mode,
    }


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------
FAILURE_FIELDS = [
    "source_file", "row_id", "raw_smiles", "failure_reason", "exception_str"
]


def process_dataset(
    df: pd.DataFrame,
    smiles_col: str,
    id_col: str,
    name_col: str,
    source_label: str,
    cfg: dict,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Canonicalize one dataset's SMILES column.

    Returns (output_df, failure_rows).
    """
    log_every = cfg.get("log_every_n_rows", 500)
    total = len(df)
    out_rows = []
    failure_rows = []

    t0 = time.perf_counter()
    ok_count = 0
    fail_count = 0

    for i, (_, row) in enumerate(df.iterrows()):
        raw_smiles = row.get(smiles_col, "")
        row_id = str(row.get(id_col, i))
        name = str(row.get(name_col, ""))

        result = canonicalize_smiles(
            raw_smiles, source_label, row_id, cfg, failure_rows
        )

        if result is None:
            fail_count += 1
            out_rows.append({
                "source_id":                row_id,
                "source_name":              name,
                "original_smiles":          str(raw_smiles)[:300] if raw_smiles else "",
                "canonical_smiles":         None,
                "canonical_isomeric_smiles": None,
                "inchi":                    None,
                "inchi_key":                None,
                "stereo_status":            None,
                "mol_weight":               None,
                "tautomer_mode":            cfg.get("tautomer_mode", "off"),
                "canonicalized_ok":         False,
                "failure_reason":           failure_rows[-1]["failure_reason"] if failure_rows else "unknown",
            })
        else:
            ok_count += 1
            out_rows.append({
                "source_id":                row_id,
                "source_name":              name,
                "original_smiles":          str(raw_smiles)[:300] if raw_smiles else "",
                **result,
                "canonicalized_ok":         True,
                "failure_reason":           "",
            })

        # Performance logging
        if (i + 1) % log_every == 0 or (i + 1) == total:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = total - (i + 1)
            eta_s = remaining / rate if rate > 0 else 0
            logger.info(
                f"  [{source_label}] {i+1}/{total} | "
                f"{rate:.0f} mol/s | ETA {eta_s:.0f}s | "
                f"ok={ok_count} fail={fail_count}"
            )

    return pd.DataFrame(out_rows), failure_rows


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase1_chem")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"
    )
    fh = logging.FileHandler(log_dir / "phase1_chem.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="IFG-26 Phase 1 Chemical Canonicalization")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="Path to YAML config (default: configs/experiment/phase1_default.yaml)"
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[FATAL] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        cfg_full = yaml.safe_load(f)

    chem_cfg = cfg_full.get("canonicalization", {})
    inp_cfg  = cfg_full.get("inputs", {})
    out_cfg  = cfg_full.get("outputs", {})

    # ── Setup paths ───────────────────────────────────────────────────
    out_base = ROOT / out_cfg.get("base_dir", "dataset/phase1")
    out_base.mkdir(parents=True, exist_ok=True)
    log_dir  = ROOT / "logs"
    logger   = setup_logging(log_dir)

    ts = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 70)
    logger.info(f"IFG-26 Phase 1 — Chemical Canonicalization  {ts}")
    logger.info(f"Config:         {config_path}")
    logger.info(f"tautomer_mode:  {chem_cfg.get('tautomer_mode', 'off')}")
    logger.info(f"largest_frag:   {chem_cfg.get('largest_fragment', True)}")
    logger.info(f"uncharge:       {chem_cfg.get('uncharge', True)}")
    logger.info("=" * 70)

    all_failures: list[dict] = []
    audit: dict = {
        "run_timestamp": ts,
        "config_path": str(config_path),
        "tautomer_mode": chem_cfg.get("tautomer_mode", "off"),
        "assumptions": [
            "tautomer_mode=off: InChIKey via standardized InChI handles tautomer-invariant dedup",
            "MGDB skiprows=1 to skip BOM + literal row-0 artefact",
            "Protein sequences in SMILES column classified as protein_sequence_contamination",
            "Markush / wildcard SMILES (*) classified as markush_star and skipped",
            "stereo_status uses FindPotentialStereo; no stereo imputation performed",
        ],
        "datasets": {}
    }

    # ── Process MGDB ──────────────────────────────────────────────────
    mgdb_cfg = inp_cfg.get("mgdb_compounds", {})
    mgdb_path = ROOT / mgdb_cfg.get("path", "data/raw/mgdb/mgdb_compounds.csv")
    logger.info(f"\nLoading MGDB: {mgdb_path}")
    mgdb_df = pd.read_csv(
        mgdb_path,
        skiprows=mgdb_cfg.get("skiprows", 1),
        encoding=mgdb_cfg.get("encoding", "utf-8-sig"),
        low_memory=False,
    )
    logger.info(f"  Loaded {len(mgdb_df)} rows | "
                f"SMILES col '{mgdb_cfg.get('smiles_col','Smiles')}' "
                f"non-null: {mgdb_df[mgdb_cfg.get('smiles_col','Smiles')].notna().sum()}")

    t_mgdb = time.perf_counter()
    mgdb_out, mgdb_fail = process_dataset(
        mgdb_df,
        smiles_col=mgdb_cfg.get("smiles_col", "Smiles"),
        id_col=mgdb_cfg.get("id_col", "ID"),
        name_col=mgdb_cfg.get("name_col", "Name"),
        source_label="mgdb_compounds",
        cfg=chem_cfg,
        logger=logger,
    )
    mgdb_elapsed = time.perf_counter() - t_mgdb
    all_failures.extend(mgdb_fail)

    # MGDB stats
    mgdb_ok = mgdb_out["canonicalized_ok"].sum()
    mgdb_fail_n = len(mgdb_out) - mgdb_ok
    mgdb_fail_reasons = (
        mgdb_out[~mgdb_out["canonicalized_ok"]]["failure_reason"]
        .value_counts().to_dict()
    )
    logger.info(
        f"\n  MGDB summary: {len(mgdb_out)} rows | "
        f"ok={mgdb_ok} ({100*mgdb_ok/len(mgdb_out):.1f}%) | "
        f"fail={mgdb_fail_n} | elapsed={mgdb_elapsed:.1f}s "
        f"({len(mgdb_out)/mgdb_elapsed:.0f} mol/s)"
    )
    logger.info(f"  Failure breakdown: {mgdb_fail_reasons}")

    audit["datasets"]["mgdb_compounds"] = {
        "total_rows": len(mgdb_out),
        "canonicalized_ok": int(mgdb_ok),
        "failed": int(mgdb_fail_n),
        "ok_rate_pct": round(100 * mgdb_ok / len(mgdb_out), 2),
        "elapsed_s": round(mgdb_elapsed, 2),
        "mol_per_sec": round(len(mgdb_out) / mgdb_elapsed, 1),
        "failures_by_reason": mgdb_fail_reasons,
    }

    # ── Process MGTbind ───────────────────────────────────────────────
    mgt_cfg = inp_cfg.get("mgtbind_compounds", {})
    mgt_path = ROOT / mgt_cfg.get("path", "data/raw/mgtbind/mgtbind_compounds.csv")
    logger.info(f"\nLoading MGTbind: {mgt_path}")
    mgt_df = pd.read_csv(
        mgt_path,
        skiprows=mgt_cfg.get("skiprows", 0),
        encoding=mgt_cfg.get("encoding", "utf-8"),
        low_memory=False,
    )
    logger.info(f"  Loaded {len(mgt_df)} rows | "
                f"SMILES col '{mgt_cfg.get('smiles_col','canonical_smiles')}' "
                f"non-null: {mgt_df[mgt_cfg.get('smiles_col','canonical_smiles')].notna().sum()}")

    t_mgt = time.perf_counter()
    mgt_out, mgt_fail = process_dataset(
        mgt_df,
        smiles_col=mgt_cfg.get("smiles_col", "canonical_smiles"),
        id_col=mgt_cfg.get("id_col", "id"),
        name_col=mgt_cfg.get("name_col", "name"),
        source_label="mgtbind_compounds",
        cfg=chem_cfg,
        logger=logger,
    )
    mgt_elapsed = time.perf_counter() - t_mgt
    all_failures.extend(mgt_fail)

    mgt_ok = mgt_out["canonicalized_ok"].sum()
    mgt_fail_n = len(mgt_out) - mgt_ok
    mgt_fail_reasons = (
        mgt_out[~mgt_out["canonicalized_ok"]]["failure_reason"]
        .value_counts().to_dict()
    )
    logger.info(
        f"\n  MGTbind summary: {len(mgt_out)} rows | "
        f"ok={mgt_ok} ({100*mgt_ok/len(mgt_out):.1f}%) | "
        f"fail={mgt_fail_n} | elapsed={mgt_elapsed:.1f}s "
        f"({len(mgt_out)/mgt_elapsed:.0f} mol/s)"
    )
    logger.info(f"  Failure breakdown: {mgt_fail_reasons}")

    audit["datasets"]["mgtbind_compounds"] = {
        "total_rows": int(len(mgt_out)),
        "canonicalized_ok": int(mgt_ok),
        "failed": int(mgt_fail_n),
        "ok_rate_pct": round(100 * mgt_ok / len(mgt_out), 2),
        "elapsed_s": round(mgt_elapsed, 2),
        "mol_per_sec": round(len(mgt_out) / mgt_elapsed, 1),
        "failures_by_reason": mgt_fail_reasons,
    }

    # ── Write outputs ─────────────────────────────────────────────────
    mgdb_out_path = out_base / out_cfg.get("mgdb_out", "mgdb_compounds_canonicalized.csv")
    mgt_out_path  = out_base / out_cfg.get("mgtbind_out", "mgtbind_compounds_canonicalized.csv")
    fail_path     = out_base / out_cfg.get("failures_out", "chem_failures.csv")
    audit_path    = out_base / out_cfg.get("audit_out", "phase1_chem_audit.json")

    mgdb_out.to_csv(mgdb_out_path, index=False, encoding="utf-8")
    logger.info(f"\n  Written: {mgdb_out_path}")

    mgt_out.to_csv(mgt_out_path,  index=False, encoding="utf-8")
    logger.info(f"  Written: {mgt_out_path}")

    # Failures
    if all_failures:
        fail_df = pd.DataFrame(all_failures, columns=FAILURE_FIELDS)
        fail_df.to_csv(fail_path, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=FAILURE_FIELDS).to_csv(fail_path, index=False, encoding="utf-8")
    logger.info(f"  Written: {fail_path}  ({len(all_failures)} failure records)")

    # Audit JSON
    total_ok   = int(mgdb_ok + mgt_ok)
    total_rows = len(mgdb_out) + len(mgt_out)
    audit["totals"] = {
        "total_rows": total_rows,
        "canonicalized_ok": total_ok,
        "failed": total_rows - total_ok,
        "ok_rate_pct": round(100 * total_ok / total_rows, 2),
        "total_failures_rows": len(all_failures),
    }

    # Top failure examples (first 3 of each reason)
    from collections import defaultdict
    top_ex: dict[str, list] = defaultdict(list)
    for f in all_failures:
        r = f["failure_reason"]
        if len(top_ex[r]) < 3:
            top_ex[r].append({
                "source": f["source_file"], "id": f["row_id"],
                "smiles_preview": f["raw_smiles"][:60],
            })
    audit["top_failure_examples"] = dict(top_ex)

    if chem_cfg.get("write_audit", True):
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        logger.info(f"  Written: {audit_path}")

    # ── Final summary ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("IFG-26 Phase 1 Chemical Canonicalization — COMPLETE")
    logger.info(f"  Total rows processed : {total_rows}")
    logger.info(f"  Canonicalized OK     : {total_ok} ({100*total_ok/total_rows:.1f}%)")
    logger.info(f"  Failed               : {total_rows - total_ok}")
    logger.info(f"  MGDB speed           : {audit['datasets']['mgdb_compounds']['mol_per_sec']} mol/s")
    logger.info(f"  MGTbind speed        : {audit['datasets']['mgtbind_compounds']['mol_per_sec']} mol/s")
    logger.info(f"  tautomer_mode        : {chem_cfg.get('tautomer_mode', 'off')} (Phase 1 default)")
    logger.info("=" * 70)
    logger.info("\nNext step: python scripts/phase1_merge_pairs.py")


if __name__ == "__main__":
    main()
