"""
phase5_generate_figures.py
============================
IFG-26 Phase 5 — Performance Collapse Visualization.

Reads phase5_negative_ladder_results.csv and generates
the negative realism ladder figure.

Outputs:
    figures/negative_realism_ladder.png

Usage:
    python scripts/phase5_generate_figures.py [--resume]
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore")

TIERS = ["Proxy", "PMD-v1", "PMD-strict", "PMD-v2", "PU-pool"]
TIER_LABELS = ["Proxy", "PMD-v1", "PMD-Strict", "PMD-v2", "PU Pool"]
MODEL_COLORS = {"LR": "#1976D2", "RF": "#E53935"}
FEAT = "ECFP4"  # primary feature set for the ladder figure


def setup_logging():
    lg = logging.getLogger("phase5_figures")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"))
    lg.addHandler(sh)
    return lg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lg = setup_logging()

    fig_path = ROOT / "figures" / "negative_realism_ladder.png"
    if args.resume and fig_path.exists():
        lg.info("negative_realism_ladder.png already exists — skipping (--resume).")
        return

    results_path = ROOT / "data" / "phase5_negative_ladder_results.csv"
    if not results_path.exists():
        lg.error("phase5_negative_ladder_results.csv not found. Run phase5_negative_ladder_eval.py first.")
        sys.exit(1)

    df = pd.read_csv(results_path)
    df = df[df["feature_set"] == FEAT]

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("auroc",          "AUROC",          "AUROC (5-fold CV)"),
        ("recall_at_5pct", "Recall@5%",      "Recall @ Top 5%"),
        ("lift_at_5pct",   "Lift@5%",        "Lift @ Top 5%"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("IFG-26 — Negative Realism Ladder (ECFP4 features)", fontsize=14, fontweight="bold")

    x = np.arange(len(TIERS))

    for ax, (col, label, title) in zip(axes, metrics):
        for mdl, color in MODEL_COLORS.items():
            vals = []
            for tier in TIERS:
                sub = df[(df["tier"] == tier) & (df["model"] == mdl)]
                vals.append(float(sub[col].values[0]) if not sub.empty else np.nan)
            ax.plot(x, vals, marker="o", label=mdl, color=color, linewidth=2, markersize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(TIER_LABELS, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(label)
        ax.set_xlabel("Negative Set")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    lg.info(f"Written: {fig_path.relative_to(ROOT)}")
    lg.info("Phase 5 Figures — COMPLETE")


if __name__ == "__main__":
    main()
