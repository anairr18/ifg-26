import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent

def fit_bootstrapped_scaling(df, n_iterations=100, seed=42):
    # Prepare hits and similarity
    # We use a global 5% threshold from the combined P+U pool
    scores = df['score'].values
    labels = (df['source'] != 'u_pool').astype(int)
    
    n_tot = len(scores)
    k_thresh_idx = max(int(n_tot * 0.05) - 1, 0)
    sorted_sc = np.sort(scores)[::-1]
    threshold_5 = sorted_sc[k_thresh_idx]
    
    # Hits among actual positives
    df_P = df[df['source'] != 'u_pool'].copy()
    hit_mask = (df_P['score'].values >= threshold_5).astype(int)
    nn_sim = df_P['nn_to_train'].values
    
    sim_grid = np.linspace(0.0, 1.0, 100)
    preds = []
    
    rng = np.random.RandomState(seed)
    
    # Base fit
    lr = LogisticRegression(penalty='l2', C=1e10)
    lr.fit(nn_sim.reshape(-1, 1), hit_mask)
    base_curve = lr.predict_proba(sim_grid.reshape(-1, 1))[:, 1]
    
    for i in range(n_iterations):
        # Bootstrap resampling
        idx = rng.choice(len(hit_mask), len(hit_mask), replace=True)
        X_res = nn_sim[idx].reshape(-1, 1)
        y_res = hit_mask[idx]
        
        # Check if we have both classes
        if len(np.unique(y_res)) < 2:
            continue
            
        m = LogisticRegression(penalty='l2', C=1e10)
        m.fit(X_res, y_res)
        preds.append(m.predict_proba(sim_grid.reshape(-1, 1))[:, 1])
        
    preds = np.array(preds)
    lo = np.percentile(preds, 2.5, axis=0) if len(preds) > 0 else base_curve
    hi = np.percentile(preds, 97.5, axis=0) if len(preds) > 0 else base_curve
    
    return sim_grid, base_curve, lo, hi

def main():
    print("Executing Figure Hardening (Phase D)...")
    
    pred_dir = ROOT / 'data/preds/phase5'
    
    # Load Scaffold Test
    df_scaff = pd.read_parquet(pred_dir / 'nnpu_LP0_scaffold.parquet')
    df_scaff = df_scaff[df_scaff['subset'] == 'test']
    
    # Load Target Holdout
    df_th = pd.read_parquet(pred_dir / 'nnpu_LP0_target_holdout.parquet')
    # th parquet is already subset to test in precompute script
    
    # 1. Scaling fits
    grid, curve_sc, lo_sc, hi_sc = fit_bootstrapped_scaling(df_scaff)
    _, curve_th, lo_th, hi_th = fit_bootstrapped_scaling(df_th)
    
    # 2. Plotting
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    
    # Scaffold Plot
    plt.plot(grid, curve_sc, color='#1f77b4', linewidth=3, label='Interaction Model (Scaffold Test)')
    plt.fill_between(grid, lo_sc, hi_sc, color='#1f77b4', alpha=0.2, label='95% Bootstrap CI (Scaffold)')
    
    # Target Holdout Plot
    plt.plot(grid, curve_th, color='#ff7f0e', linewidth=3, linestyle='--', label='Interaction Model (Target Holdout)')
    plt.fill_between(grid, lo_th, hi_th, color='#ff7f0e', alpha=0.1, label='95% Bootstrap CI (Target Holdout)')
    
    # Analog Horizon Marking
    plt.axvline(x=0.4, color='red', linestyle=':', linewidth=2, label='Analog Horizon (0.4 Tanimoto)')
    plt.text(0.41, 0.02, 'Hard Generalization Frontier', color='red', weight='bold', verticalalignment='bottom')
    
    # Labels and Formatting
    plt.xlabel('Tanimoto Similarity to Training Set (Max NN)', fontsize=12)
    plt.ylabel('PU-Recall @ 5% Probability', fontsize=12)
    plt.title('IFG-26 Scaling Law: Generalization vs. Structural Distance', fontsize=14, pad=15)
    plt.xlim(0, 1)
    plt.ylim(0, max(hi_sc) * 1.1)
    
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    
    # Branding
    plt.text(0.98, 0.02, 'IFG-26 Benchmark v1.2', horizontalalignment='right', 
             verticalalignment='bottom', transform=plt.gca().transAxes, alpha=0.5, fontsize=10)
    
    # Save
    out_dir = ROOT / 'results/figures'
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / 'similarity_scaling_v2.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[DONE] High-resolution figure saved to {out_path}")

if __name__ == '__main__':
    main()
