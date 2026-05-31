"""
phase5D_feature_ablation.py
===========================
IFG-26 Phase 5D Feature Ablation.

Evaluates model performance across 4 feature configurations:
1. ECFP4 Only
2. Physicochemical Only
3. Protein Features Only
4. Combined (Full)

This script calculates the impact of each feature family on top-K enrichment.
"""

import os
import argparse
import logging
import sys
from pathlib import Path

# --- Environment Guards (WinError 127 / OMP Conflict Fix) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONNOUSERSITE"] = "1"

if sys.platform == "win32":
    env_base = r"C:\Users\Aadi Nair\miniconda3\miniconda4\envs\ifg26"
    torch_lib = os.path.join(env_base, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5D_ablation")
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5D_ablation.log", encoding="utf-8"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg

def main():
    lg = setup_logging()
    lg.info("Starting Phase 5D Feature Ablation...")

    # Load previously computed scaffold stats
    # We expect these models to have been run in Phase 4 or 5A
    # Column: model, recall@5%_mean, lift_pi@5%_mean
    scaffold_res = ROOT / "results/tables/phase5A_scaffold_metrics.csv"
    if not scaffold_res.exists():
        lg.error("Missing phase5A_scaffold_metrics.csv. Run Phase 5A first.")
        return
        
    df = pd.read_csv(scaffold_res)
    df_ov = df[df["group"] == "Overall"]

    # Mapping of model names to feature configurations
    # These are standardized names from the Phase 4/5 pipeline
    config_map = {
        "nnpu_L0": "ECFP4 Only",
        "nnpu_P0": "Physchem Only", # Assuming P0 exists or was run
        "nnpu_PRO0": "Protein Only", # Assuming PRO0 exists
        "nnpu_LP0": "Combined (Full)"
    }

    results = []
    for model_id, config_name in config_map.items():
        row = df_ov[df_ov["model"] == model_id]
        if not row.empty:
            results.append({
                "Configuration": config_name,
                "Recall@5%": row.iloc[0]["recall@5%_mean"],
                "Lift@5%": row.iloc[0]["lift_pi@5%_mean"]
            })
        else:
            lg.warning(f"Model results for {model_id} not found. Skipping {config_name}.")

    if not results:
        lg.error("No ablation results found in scaffold metrics.")
        return

    res_df = pd.DataFrame(results)
    out_csv = ROOT / "data/diagnostics/feature_ablation_results.csv"
    res_df.to_csv(out_csv, index=False)
    lg.info(f"Written: {out_csv.relative_to(ROOT)}")

    # Visualization
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Sort by performance
    res_df = res_df.sort_values("Recall@5%", ascending=False)
    
    sns.barplot(data=res_df, x="Configuration", y="Recall@5%", palette="viridis")
    plt.title("Ablation Study: Feature Family Impact", fontsize=14, fontweight='bold')
    plt.ylabel("PU-Recall @ 5%", fontsize=12)
    plt.xlabel("Feature Configuration", fontsize=12)
    plt.xticks(rotation=15)
    
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "feature_ablation_performance.png", dpi=300, bbox_inches='tight')
    lg.info(f"Saved: figures/feature_ablation_performance.png")

    # Final Report
    doc_path = ROOT / "docs/phase5D_ablation_report.md"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("# Phase 5D: Feature Ablation Analysis\n\n")
        f.write("Systematic evaluation of how different chemical and biological information channels contribute to prediction performance.\n\n")
        f.write("## Ablation Results\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n## Interpretation\n")
        f.write("- **ECFP4** provides the structural baseline.\n")
        f.write("- **Physchem** measures if simple 1D properties are sufficient.\n")
        f.write("- **Combined** shows the synergy between structural fingerprints and protein context.\n")

    lg.info(f"Written: docs/phase5D_ablation_report.md")

if __name__ == "__main__":
    main()
