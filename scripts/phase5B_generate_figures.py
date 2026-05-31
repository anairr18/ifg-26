"""
phase5B_generate_figures.py
==============================
IFG-26 Phase 5B — Performance Collapse Visualization.

Reads phase5B_negative_ladder_results.csv and generates the
"Negative Realism Ladder" figure, testing the hypothesis that performance
collapses as negative difficulty increases.

Outputs:
    figures/negative_realism_ladder_external.png
"""

import argparse
import logging
import os
import sys
import warnings
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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))

warnings.filterwarnings("ignore")

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("phase5B_figures")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    lg.addHandler(logging.FileHandler(log_dir / "phase5B_figures.log", encoding="utf-8"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg

def main():
    lg = setup_logging()
    lg.info("Generating Phase 5B figures...")

    res_path = ROOT / "data" / "phase5B_negative_ladder_results.csv"
    if not res_path.exists():
        lg.error("Ladder results not found. Run phase5B_negative_ladder.py first.")
        sys.exit(1)

    df = pd.read_csv(res_path)

    # Order tiers by realism
    tier_order = ["Proxy", "PMD-v1", "Strict PMD", "External PMD", "PU Pool"]
    df["tier"] = pd.Categorical(df["tier"], categories=tier_order, ordered=True)
    df = df.sort_values("tier")

    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))

    metrics = ["auroc", "auprc"]
    colors = {"LR": "#3498db", "RF": "#e74c3c", "nnPU_L0": "#2ecc71", "nnPU_LP0": "#9b59b6"}

    for model in df["model"].unique():
        m_df = df[df["model"] == model]
        plt.plot(m_df["tier"], m_df["auroc"], marker='o', label=f"{model} (AUROC)", 
                 color=colors.get(model, "#95a5a6"), linewidth=2, markersize=8)
        
    plt.ylim(0.45, 1.05)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label="Random Baseline")
    
    plt.title("IFG-26 Negative Realism Ladder — External Universe", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Negative Realism Tier", fontsize=12, labelpad=10)
    plt.ylabel("Performance (AUROC)", fontsize=12, labelpad=10)
    plt.legend(frameon=True, loc='lower left')
    
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "negative_realism_ladder_external.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    lg.info(f"Saved: {out_path.relative_to(ROOT)}")

    # Secondary Plot: Recall@5
    plt.figure(figsize=(10, 6))
    for model in df["model"].unique():
        m_df = df[df["model"] == model]
        plt.plot(m_df["tier"], m_df["recall@5"], marker='s', label=f"{model} (Recall@5)", 
                 color=colors.get(model, "#95a5a6"), linewidth=2, markersize=8)
                 
    plt.title("IFG-26 High-Precision Performance", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Negative Realism Tier", fontsize=12, labelpad=10)
    plt.ylabel("Recall @ Top 5%", fontsize=12, labelpad=10)
    plt.legend(frameon=True, loc='upper right')
    
    out_recall = fig_dir / "high_precision_collapse_external.png"
    plt.tight_layout()
    plt.savefig(out_recall, dpi=300)
    lg.info(f"Saved: {out_recall.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
