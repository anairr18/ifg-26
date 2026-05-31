"""
run_phase4_full.py
==================
IFG-26 — Phase 4 Master Orchestrator

Executes Phase 4 A->F sequentially.
Supports resume and graceful ML fallback dependencies.

Usage:
    python scripts/run_phase4_full.py [--resume] [--from-step STEP] [--skip-optional-models]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase4_orchestrator")
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(log_dir / "phase4_run_full.log", encoding="utf-8")
    fh.setFormatter(fmt); lg.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); lg.addHandler(sh)
    return lg

def run_step(lg, step_id, script_path, description, args=None, allow_failure=False):
    lg.info("\n" + "="*80)
    lg.info(f"STEP {step_id}: {description}")
    lg.info(f"Executing: python {script_path}")
    lg.info("="*80)
    
    cmd = [sys.executable, str(ROOT / script_path)]
    if args:
        cmd.extend(args)
        
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        lg.info(proc.stdout)
        if proc.stderr:
            lg.warning(proc.stderr)
        lg.info(f"✅ STEP {step_id} SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        lg.error(f"❌ STEP {step_id} FAILED with exit code {e.returncode}")
        lg.error(e.stdout)
        lg.error(e.stderr)
        if not allow_failure:
            lg.error("ABORTING PIPELINE.")
            sys.exit(1)
        else:
            lg.warning("Continuing pipeline despite failure (allow_failure=True).")
            return False

def check_outputs_exist(step_id):
    """Simple check for resume mode to see if we can skip."""
    checks = {
        "4A": ROOT / "data/pu/phase4A_pool_drift_report.json",
        "4C": ROOT / "data/diagnostics/pmd_match_table.parquet",
        "4E": ROOT / "data/negatives/decoy_pool.parquet",
    }
    if step_id in checks:
        return checks[step_id].exists()
    return False

def write_status(lg, success, step_failed=None):
    status_file = ROOT / "data/diagnostics/phase4_status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    soft_fails = []
    warnings = []
    
    # Check 4C status
    phase4c_sig = ROOT / "data/diagnostics/phase4C_scientific_status.json"
    if phase4c_sig.exists():
        with open(phase4c_sig) as f:
            d = json.load(f)
            if not d.get("physchem_pass", True): soft_fails.append("4C Physchem AUROC exceeded")
            if not d.get("ecfp4_pass", True): soft_fails.append("4C ECFP4 AUROC exceeded")
            if not d.get("count_pass", True): soft_fails.append("4C PMD count insufficient")
            if not d.get("pmd_generation_success", True): soft_fails.append("4C generation missed target")

    # Check 4E audit status
    phase4e_sig = ROOT / "data/diagnostics/phase4E_negative_audit_status.json"
    if phase4e_sig.exists():
        with open(phase4e_sig) as f:
            d = json.load(f)
            if not d.get("physchem_pass", True): soft_fails.append("4E Physchem AUROC exceeded")
            if not d.get("ecfp4_pass", True): soft_fails.append("4E ECFP4 AUROC exceeded")
            for w in d.get("warnings", []):
                warnings.append(f"4E: {w}")
                
    hard_fail = not success
    has_soft_fail = len(soft_fails) > 0
    
    if hard_fail:
        scientific_status = "fail"
    elif has_soft_fail:
        scientific_status = "fail"
    elif len(warnings) > 0:
        scientific_status = "warning"
    else:
        scientific_status = "pass"
    
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "4",
        "success": success and not has_soft_fail,
        "hard_fail": hard_fail,
        "soft_fail": has_soft_fail,
        "scientific_status": scientific_status,
        "failed_at_step": step_failed,
        "soft_failure_reasons": soft_fails,
        "warnings": warnings,
        "output_paths": {
            "p_pool": "data/pu/pool_P_scaffold.parquet",
            "u_pool": "data/pu/pool_U_scaffold.parquet",
            "pmd": "data/negatives/pmd_negatives.parquet",
            "decoy": "data/negatives/decoy_pool.parquet"
        }
    }
    with open(status_file, "w") as f:
        json.dump(data, f, indent=2)
    lg.info(f"Status written to {status_file.relative_to(ROOT)}. Hard Fail: {hard_fail}, Soft Fails: {len(soft_fails)}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Automatically skip steps whose outputs already exist")
    parser.add_argument("--from-step", type=str, help="Start directly from a specific step (e.g., 4C)")
    parser.add_argument("--skip-optional-models", action="store_true", help="Force skip XGBoost/LightGBM runs if known missing")
    parser.add_argument("--smoke", action="store_true", help="Pass --smoke down to applicable scripts (Phase 4E)")
    args = parser.parse_args()

    lg = setup_logging()
    
    lg.info("Starting Phase 4 Pipeline Orchestrator")
    
    # Pre-flight dependencies
    run_step(lg, "PRE", "scripts/check_phase4_dependencies.py", "Dependency Check")
    
    steps = [
        ("4A", "scripts/phase4_build_pu_pools.py", "PU Pool Construction"),
        ("4C", "scripts/phase4_generate_pmd_negatives.py", "PMD Negative Generation"),
        ("4C-ii", "scripts/phase4_binary_eval.py", "Binary Artifact Audit"),
        ("4B", "scripts/phase4_train_nnpu.py", "nnPU Model Training"), # If this exists, else ignore
        ("4D", "scripts/phase4_compute_protein_embeddings.py", "ESM2 Protein Embeddings"), # Optional usually
        ("4E", "scripts/phase4E_expand_decoy_pool.py", "Decoy Expansion"),
        ("4F", "scripts/phase4E_generate_strict_pmd.py", "Relaxation Ladder PMD"),
        ("4G", "scripts/phase4E_negative_audit.py", "Phase 4E Final Audit"),
        ("4H", "scripts/phase4E_similarity_stratified_eval.py", "Similarity Stratified Integrity Eval")
    ]
    
    skip_until = args.from_step if args.from_step else None
    
    for step_id, script, desc in steps:
        if not (ROOT / script).exists():
            lg.warning(f"Skipping {step_id}: Script {script} not found in this version of the codebase.")
            continue
            
        if skip_until:
            if step_id == skip_until:
                skip_until = None  # Start here
            else:
                lg.info(f"Skipping {step_id} (waiting for --from-step {skip_until})")
                continue
                
        if args.resume and check_outputs_exist(step_id):
            lg.info(f"Skipping {step_id} (Outputs already exist, --resume active)")
            continue
            
        step_args = []
        if step_id == "4E" and args.smoke:
            step_args.append("--smoke")
            
        success = run_step(lg, step_id, script, desc, args=step_args)
        if not success:
            lg.error(f"HARD FAILURE at Step {step_id}.")
            write_status(lg, False, step_failed=step_id)
            return

    final_status = write_status(lg, True)
    
    # Generate final run summary md
    status_str = "SUCCESS" if final_status["success"] else ("SOFT SCIENTIFIC FAILURES" if final_status["soft_fail"] else "HARD FAILURE")
    sci_stat = final_status.get("scientific_status", "unknown").upper()
    
    soft_str = "\n".join([f"- {s}" for s in final_status['soft_failure_reasons']]) if final_status['soft_failure_reasons'] else "None"
    warn_str = "\n".join([f"- {w}" for w in final_status['warnings']]) if final_status['warnings'] else "None"
    
    summary = f"""# IFG-26 Phase 4 Full Run Summary

_Generated: {datetime.now(timezone.utc).isoformat()}_

## Pipeline Execution Status: **{status_str}**
## Scientific Status: **{sci_stat}**

The Phase 4 orchestrator `run_phase4_full.py` executed scripts sequentially.

### Hard Failures
{"None" if not final_status['hard_fail'] else f"Failed at {final_status['failed_at_step']}"}

### Soft Scientific Failures
{soft_str}

### Warnings
{warn_str}

| Phase | Description |
|-------|-------------|
| **4A**  | PU Pool Construction |
| **4B**  | nnPU Model Training |
| **4C**  | PMD Generation |
| **4C-ii**| Binary Artifact Eval |
| **4D**  | ESM2 Embeddings |
| **4E**  | Decoy Expansion |
| **4F**  | Strict PMD Ranking |
| **4G**  | Phase 4E Final Audit |
"""
    with open(ROOT / "docs" / "phase4_run_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)
    lg.info("Written: docs/phase4_run_summary.md")
    lg.info(f"Phase 4 Pipeline Complete. Final state: {status_str}")

if __name__ == "__main__":
    main()
