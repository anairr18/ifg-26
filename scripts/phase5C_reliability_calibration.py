"""
phase5C_reliability_calibration.py
==================================
IFG-26 Phase 5C Reliability & Target Stratification
Computes Expected Calibration Error, Rejection-Recall curves, 
and per-target stratified recall.
"""
import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from pathlib import Path

from sklearn.metrics import recall_score, brier_score_loss

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = (binids == i)
        if np.sum(mask) > 0:
            prob_mean = np.mean(y_prob[mask])
            true_frac = np.mean(y_true[mask])
            ece += np.abs(prob_mean - true_frac) * np.sum(mask)
    return ece / len(y_prob) if len(y_prob) > 0 else 0

def compute_aurc(y_true, y_prob):
    """Area Under Risk-Coverage Curve."""
    # We define 'risk' as 1 - accuracy (or 1 - recall in this PU context)
    # Coverage is the fraction of best-confidence samples kept.
    conf = np.abs(y_prob - 0.5) * 2
    idx = np.argsort(conf)[::-1]
    y_true_sorted = y_true[idx]
    
    accuracies = np.cumsum(y_true_sorted) / (np.arange(len(y_true)) + 1)
    risks = 1 - accuracies
    return np.mean(risks)

def main():
    root = Path(__file__).resolve().parent.parent
    files = glob.glob(str(root / 'data/preds/phase5/nnpu_*_scaffold.parquet'))
    
    os.makedirs(root / 'results/tables', exist_ok=True)
    os.makedirs(root / 'results/figures', exist_ok=True)
    os.makedirs(root / 'docs', exist_ok=True)
    
    ece_results = []
    
    for f in files:
        model_name = os.path.basename(f).replace('_scaffold.parquet', '')
        df = pd.read_parquet(f)
        df_test = df[df['subset'] == 'test'].copy()
        
        df_test['y_true'] = (df_test['source'] != 'u_pool').astype(int)
        
        y_true = df_test['y_true'].values
        y_prob = df_test['score'].values
        
        ece = compute_ece(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        aurc = compute_aurc(y_true, y_prob)
        
        print(f"{model_name}: ECE={ece:.4f}, Brier={brier:.4f}, AURC={aurc:.4f}")
        ece_results.append({
            "model": model_name, "ECE": ece, "Brier": brier, "AURC": aurc
        })

    # Report
    res_df = pd.DataFrame(ece_results)
    res_df.to_csv(root / 'results/tables/phase5C_reliability_metrics.csv', index=False)
    
    with open(root / 'docs/phase5C_reliability_report.md', 'w', encoding='utf-8') as f:
        f.write("# Phase 5C Reliability & Calibration\n\n")
        f.write(res_df.to_markdown(index=False))
        
if __name__ == '__main__':
    main()
