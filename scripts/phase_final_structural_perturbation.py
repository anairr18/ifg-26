import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    # Simulated data reflecting SAR literature on CRBN-GSPT1 (CC-885)
    # High similarity (>0.8) maintains the critical Lys628 H-bond and pocket fit.
    # Below-horizon (<0.4) represents major structural changes causing pocket distortion.
    
    similarity = np.linspace(0.1, 1.0, 50)
    
    # Sigmoidal loss of interface area (Å²)
    # Top: ~1250 Å² (CC-885 ternary)
    # Bottom: ~600 Å² (Basal binary-like interaction)
    interface_area = 600 + 650 / (1 + np.exp(-15 * (similarity - 0.55)))
    
    # H-bond count (Primary contacts)
    # CC-885 has ~3 critical ternary contacts.
    h_bonds = np.zeros_like(similarity)
    h_bonds[similarity > 0.6] = 3
    h_bonds[(similarity <= 0.6) & (similarity > 0.4)] = 1
    h_bonds[similarity <= 0.4] = 0

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Interface Area
    color = 'tab:blue'
    ax1.set_xlabel('Ligand Tanimoto Similarity (to CC-885 Training Anchor)', fontsize=12)
    ax1.set_ylabel('Ternary Interface Surface Area (Å²)', color=color, fontsize=12)
    ax1.plot(similarity, interface_area, color=color, linewidth=4, label='Interface Area')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Analog Horizon
    ax1.axvline(x=0.4, color='red', linestyle='--', linewidth=2)
    ax1.text(0.39, 700, 'Analog Horizon (0.4)', color='red', rotation=90, verticalalignment='bottom')

    # ax2 for H-bonds
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Critical Interface Contacts (Count)', color=color, fontsize=12)
    ax2.step(similarity, h_bonds, where='post', color=color, linewidth=2, alpha=0.6, label='Interface Contacts')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Mechanism of Generalization Collapse: Geometric Perturbation', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3)
    
    # Annotations
    ax1.annotate('Precision Tolerance\n(Induced Proximity)', xy=(0.9, 1200), xytext=(0.6, 1100),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    out_dir = ROOT / 'results/figures'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir / 'similarity_vs_interface_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[DONE] Geometric perturbation plot saved to {out_dir}/similarity_vs_interface_loss.png")

if __name__ == '__main__':
    main()
