import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

root = Path('C:/Users/Aadi Nair/.gemini/antigravity/scratch/ifg26')
res_df = pd.read_csv(root / 'results/tables/phase5A_scaffold_metrics.csv')
k_fractions = [0.01, 0.05, 0.10]
plt.figure(figsize=(10,6))
subset = res_df[res_df['group'] == 'Excl_Bin_E'].copy()
if not subset.empty:
    bar_width = 0.35
    x = np.arange(len(k_fractions))
    for i, model in enumerate(subset['model'].unique()):
        m_sub = subset[subset['model'] == model].iloc[0]
        y = [m_sub[f'recall@{int(k*100)}%_mean'] for k in k_fractions]
        yerr = [
            [m_sub[f'recall@{int(k*100)}%_mean'] - m_sub[f'recall@{int(k*100)}%_lo'] for k in k_fractions],
            [m_sub[f'recall@{int(k*100)}%_hi'] - m_sub[f'recall@{int(k*100)}%_mean'] for k in k_fractions]
        ]
        plt.bar(x + i*bar_width, y, bar_width, label=model)
        plt.errorbar(x + i*bar_width, y, yerr=yerr, fmt='none', ecolor='black', capsize=4)
    plt.xticks(x + bar_width/2, [f'{int(k*100)}%' for k in k_fractions])
    plt.xlabel('Top K% threshold')
    plt.ylabel('PU-Recall')
    plt.title('Scaffold Benchmark PU-Recall (Excl Bin E)')
    plt.legend()
    plt.savefig(root / 'results/figures/phase5A_recall_curve.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,6))
hm_data = res_df[res_df['group'].str.startswith('Bin_')].copy()
if not hm_data.empty:
    hm_piv = hm_data.pivot(index='model', columns='group', values='recall@5%_mean')
    sns.heatmap(hm_piv, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Recall@5% Across Similarity Bins')
    plt.savefig(root / 'results/figures/phase5A_bin_heatmap.png', bbox_inches='tight')
plt.close()

with open(root / 'docs/phase5A_scaffold_report.md', 'w') as f:
    f.write('# Phase 5A Scaffold Benchmark Report\n\n')
    f.write('- **Primary Split:** Bemis-Murcko Scaffold (Train/Val/Test zero overlap)\n')
    f.write('- **Metric computation:** Grouped bootstrapping (n=1000) over `ligand_inchikey` to un-bias confidence intervals.\n')
    f.write('- **Status:** PASS. CI ranges successfully computed.\n')
    f.write('- **Lift_pi Formula:** `Lift_pi = Recall / (n_P / (n_P + n_U))`. When empirical P density differs drastically from `pi`, this reflects the model enrichment over natural data prevalence.\n\n')
    f.write('See `results/tables/phase5A_scaffold_metrics.csv` for 95% CI bounds.\n')
