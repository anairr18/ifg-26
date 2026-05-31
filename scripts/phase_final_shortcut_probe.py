import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout, batch_norm):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def make_ecfp_tensor(df, ecfp_mat):
    rows = df["ligand_feature_row"].clip(0).values
    return torch.from_numpy(ecfp_mat[rows].astype(np.float32))

def make_protein_onehot(df, n_proteins):
    n_p1 = n_proteins + 1
    e3_oh = np.eye(n_p1, dtype=np.float32)[df["e3_label_id"].clip(-1).values + 1]
    tgt_oh = np.eye(n_p1, dtype=np.float32)[df["target_label_id"].clip(-1).values + 1]
    return torch.from_numpy(np.hstack([e3_oh, tgt_oh]))

def calculate_metrics(scores, labels, top_pct=0.05):
    # Standard Recall@k and Lift@k for PU evaluation
    k = int(len(scores) * top_pct)
    if k == 0: k = 1
    
    # Get top k indices
    idx = np.argsort(scores)[::-1][:k]
    tp = np.sum(labels[idx])
    total_p = np.sum(labels)
    
    recall = tp / total_p if total_p > 0 else 0
    # Lift relative to random: (tp/k) / (total_p/N)
    lift = (tp / k) / (total_p / len(scores)) if total_p > 0 else 1.0
    return recall, lift

def main():
    print("Initiating Shortcut Probing Analysis...")
    
    # 1. Load Data
    pool_P = pd.read_parquet(ROOT / 'data/pu/pool_P_scaffold.parquet')
    pool_U = pd.read_parquet(ROOT / 'data/pu/pool_U_scaffold.parquet')
    ecfp_mat = np.load(str(ROOT / 'data/features/ligands_ecfp4.npy'))
    promaps = pd.read_parquet(ROOT / 'data/features/protein_index.parquet')
    n_prot = len(promaps)
    
    # Splits
    sj = json.load(open(ROOT / 'splits/scaffold_split.json'))
    scaffold_test_idx = set(sj['test_row_indices'])
    
    # Target Holdout: We'll identify it from pool_P subset assignment logic
    # (Actually, let's just use the logic from phase5_precompute_predictions.py)
    def get_target_holdout_mask(df, seed=42):
        from random import Random
        # Cast to string and dropna for sorting
        valid_targets = df["target_uniprot"].dropna().astype(str).unique()
        unique_targets = sorted(valid_targets)
        rng = Random(seed)
        rng.shuffle(unique_targets)
        n_train = int(0.8 * len(unique_targets))
        holdout_targets = set(unique_targets[n_train:])
        return df["target_uniprot"].astype(str).isin(holdout_targets)

    # Prepare Combined Evaluation Sets
    eval_sets = {}
    
    # Scaffold Test
    P_scaff = pool_P.iloc[list(scaffold_test_idx)].copy()
    eval_sets["scaffold_test"] = pd.concat([P_scaff, pool_U], ignore_index=True)
    eval_sets["scaffold_test"]["is_pos"] = [1]*len(P_scaff) + [0]*len(pool_U)
    
    # Target Holdout
    # We need uniprot ids in pool_P. If missing, map them.
    label_to_uniprot = dict(zip(promaps['label_id'], promaps['uniprot_id']))
    if 'target_uniprot' not in pool_P.columns:
        pool_P['target_uniprot'] = pool_P['target_label_id'].map(label_to_uniprot)
    
    th_mask = get_target_holdout_mask(pool_P)
    P_th = pool_P[th_mask].copy()
    eval_sets["target_holdout"] = pd.concat([P_th, pool_U], ignore_index=True)
    eval_sets["target_holdout"]["is_pos"] = [1]*len(P_th) + [0]*len(pool_U)

    # 2. Load Models
    models = {}
    for mt in ["L0", "LP0"]:
        path = ROOT / f"results/models/nnpu_{mt}_pi0.05.pt"
        if not path.exists(): continue
        checkpoint = torch.load(path, map_location='cpu')
        m = MLP(checkpoint['in_dim'], [512,128], 0.3, True)
        m.load_state_dict(checkpoint['model_state'])
        m.eval()
        models[mt] = m

    # 3. Probing Loop
    results = []
    
    for set_name, df_eval in eval_sets.items():
        print(f"  Evaluating {set_name}...")
        ecfp = make_ecfp_tensor(df_eval, ecfp_mat)
        prot = make_protein_onehot(df_eval, n_prot)
        labels = df_eval["is_pos"].values
        
        with torch.no_grad():
            # A. Interaction (LP0)
            X_lp0 = torch.cat([ecfp, prot], dim=1)
            scores_lp0 = models["LP0"](X_lp0).numpy()
            r, l = calculate_metrics(scores_lp0, labels)
            results.append({"split": set_name, "model": "Interaction (LP0)", "recall@5%": r, "lift@5%": l})
            
            # B. Protein-only Probe (LP0 with zeroed ECFP)
            X_p_only = torch.cat([torch.zeros_like(ecfp), prot], dim=1)
            scores_p = models["LP0"](X_p_only).numpy()
            r, l = calculate_metrics(scores_p, labels)
            results.append({"split": set_name, "model": "Protein-only Probe", "recall@5%": r, "lift@5%": l})
            
            # C. Ligand-only Probe (LP0 with zeroed Protein)
            X_l_only = torch.cat([ecfp, torch.zeros_like(prot)], dim=1)
            scores_l = models["LP0"](X_l_only).numpy()
            r, l = calculate_metrics(scores_l, labels)
            results.append({"split": set_name, "model": "Ligand-only Probe", "recall@5%": r, "lift@5%": l})
            
            # D. Baseline Ligand-only (L0)
            if "L0" in models:
                scores_l0 = models["L0"](ecfp).numpy()
                r, l = calculate_metrics(scores_l0, labels)
                results.append({"split": set_name, "model": "Baseline Ligand (L0)", "recall@5%": r, "lift@5%": l})

    # 4. Save and Plot
    res_df = pd.DataFrame(results)
    os.makedirs(ROOT / 'results/tables', exist_ok=True)
    res_df.to_csv(ROOT / 'results/tables/shortcut_ablation.csv', index=False)
    print(f"[DONE] Table saved to results/tables/shortcut_ablation.csv")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=res_df, x="split", y="recall@5%", hue="model", palette="viridis")
    plt.title("Shortcut Probe: Contribution of Ligand vs Protein Features")
    plt.ylabel("Recall@5% (PU-Estimate)")
    plt.ylim(0, max(res_df["recall@5%"]) * 1.2)
    
    # Save Figure
    os.makedirs(ROOT / 'results/figures', exist_ok=True)
    plt.savefig(ROOT / 'results/figures/shortcut_ablation_barplot.png', dpi=300, bbox_inches='tight')
    print(f"[DONE] Figure saved to results/figures/shortcut_ablation_barplot.png")

if __name__ == '__main__':
    main()
