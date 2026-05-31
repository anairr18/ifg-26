"""
phase6_plus_2_calibration.py
============================
IFG-26 Additive Phase 7: Calibration Hardening

Applies Temperature Scaling to Phase 5 scaffold TEST predictions
by fitting to validation probabilities.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

def compute_ece(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = (binids == i)
        if np.sum(mask) > 0:
            prob_mean = np.mean(y_prob[mask])
            true_frac = np.mean(y_true[mask])
            ece += np.abs(prob_mean - true_frac) * np.sum(mask)
    return ece / len(y_prob)

def compute_aurc(y_true, y_prob):
    conf = np.abs(y_prob - 0.5)
    sort_idx = np.argsort(conf)[::-1]
    
    y_t = y_true[sort_idx]
    y_p = y_prob[sort_idx]
    
    errors = np.cumsum(np.round(y_p) != y_t)
    coverages = np.arange(1, len(y_t) + 1) / len(y_t)
    risks = errors / np.arange(1, len(y_t) + 1)
    
    if len(coverages) < 2: return 0.0
    # Use manual trapezoidal rule for NumPy 2.0 compatibility (trapz was removed)
    aurc = np.sum((risks[:-1] + risks[1:]) / 2 * np.diff(coverages))
    return aurc

def nll_obj(T_val, logits, labels):
    T = T_val[0]
    scaled_logits = logits / T
    # numerical stability
    probs = 1 / (1 + np.exp(-scaled_logits))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return nll

def apply_temp_scaling(scores, T):
    # recover approx logits. original scores had sigmoid applied
    eps = 1e-7
    s = np.clip(scores, eps, 1 - eps)
    logits = np.log(s / (1 - s))
    scaled = logits / T
    return 1 / (1 + np.exp(-scaled))

def main():
    root = Path(__file__).resolve().parent.parent
    # Look for Phase 5 predictions in data/preds/ or data/
    pred_path = root / 'data' / 'preds' / 'phase5' / 'nnpu_LP0_scaffold.parquet'
    if not pred_path.exists():
        fallback = list(root.glob('**/nnpu_LP0_scaffold.parquet'))
        if fallback:
            pred_path = fallback[0]
        else:
            print("Missing Phase 5 predictions (nnpu_LP0_scaffold.parquet).")
            return
        
    df = pd.read_parquet(pred_path)
    
    # 1. Isolate val/test
    # Note: U pool elements are always marked 'test' computationally.
    # To fit validation temperature scaling, we strictly use P_val against U_test (since we don't have U_val).
    # Wait, PU learning uses P vs U. Let's assign y_true = 1 for P, 0 for U.
    df['y_true'] = (df['source'] != 'u_pool').astype(int)
    
    df_val = df[(df['subset'] == 'val') | (df['source'] == 'u_pool')]
    df_test = df[(df['subset'] == 'test') | (df['source'] == 'u_pool')]
    
    val_scores = df_val['score'].values
    val_labels = df_val['y_true'].values
    
    eps = 1e-7
    v = np.clip(val_scores, eps, 1 - eps)
    val_logits = np.log(v / (1 - v))
    
    # 2. Fit Temp
    res = minimize(nll_obj, [1.0], args=(val_logits, val_labels), bounds=[(0.01, 10.0)])
    opt_T = res.x[0]
    print(f"Optimal Temperature (T): {opt_T:.4f}")
    
    # 3. Apply to test
    test_scores = df_test['score'].values
    test_labels = df_test['y_true'].values
    calib_scores = apply_temp_scaling(test_scores, opt_T)
    
    # 4. Recompute ECE, Brier, AURC
    b_ece = compute_ece(test_labels, test_scores)
    c_ece = compute_ece(test_labels, calib_scores)
    
    b_aurc = compute_aurc(test_labels, test_scores)
    c_aurc = compute_aurc(test_labels, calib_scores)
    
    b_brier = brier_score_loss(test_labels, test_scores)
    c_brier = brier_score_loss(test_labels, calib_scores)
    
    # 5. Save Calibrated Pars
    os.makedirs(root / 'data/preds/phase6_plus', exist_ok=True)
    out_df = df_test.copy()
    out_df['score'] = calib_scores
    out_df['calibrated'] = True
    out_df['T_applied'] = opt_T
    out_df.to_parquet(root / 'data/preds/phase6_plus/calibrated_nnpu_LP0_scaffold_test.parquet', index=False)
    
    # 6. Plot Side-by-Side Calibration Curve
    plt.figure(figsize=(10, 4))
    
    def plot_reliability_diagram(y_true, y_prob, ax, title):
        bins = np.linspace(0, 1, 15 + 1)
        binids = np.digitize(y_prob, bins) - 1
        bin_means = []
        true_fracs = []
        for i in range(15):
            mask = (binids == i)
            if np.sum(mask) > 0:
                bin_means.append(np.mean(y_prob[mask]))
                true_fracs.append(np.mean(y_true[mask]))
                
        ax.plot([0,1], [0,1], 'k--', label='Perfect Calibration')
        ax.plot(bin_means, true_fracs, 's-', color='red', label='Model Output')
        ax.set_title(title)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.legend()
        
    ax1 = plt.subplot(1, 2, 1)
    plot_reliability_diagram(test_labels, test_scores, ax1, f"Baseline LP0 (ECE={b_ece:.4f})")
    ax2 = plt.subplot(1, 2, 2)
    plot_reliability_diagram(test_labels, calib_scores, ax2, f"Calibrated (ECE={c_ece:.4f})")
    
    os.makedirs(root / 'results/phase6_plus', exist_ok=True)
    plt.tight_layout()
    plt.savefig(root / 'results/phase6_plus/calibration_comparison.png', bbox_inches='tight')
    plt.close()
    
    # 7. Report
    docs_dir = root / 'docs/phase6_plus'
    os.makedirs(docs_dir, exist_ok=True)
    with open(docs_dir / 'phase6_plus_calibration_report.md', 'w') as f:
        f.write("# Phase 7 Additive Extension: Calibration Hardening\n\n")
        f.write("Applying Temperature Scaling (Post-Processing) to validate reliability improvements without retraining.\n\n")
        f.write(f"- **Optimal Temperature**: {opt_T:.3f}\n\n")
        f.write("## Test Set Metrics (Scaffold Split)\n")
        f.write("| Metric | Baseline (LP0) | Calibrated (LP0) | Delta |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| ECE (15-bin) | {b_ece:.4f} | {c_ece:.4f} | {c_ece - b_ece:+.4f} |\n")
        f.write(f"| Brier Score | {b_brier:.4f} | {c_brier:.4f} | {c_brier - b_brier:+.4f} |\n")
        f.write(f"| AURC | {b_aurc:.4f} | {c_aurc:.4f} | {c_aurc - b_aurc:+.4f} |\n\n")
        f.write("![Calibration Comparison](../../results/phase6_plus/calibration_comparison.png)\n")

if __name__ == '__main__':
    main()
