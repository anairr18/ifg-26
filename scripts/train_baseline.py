"""
phase4_train_nnpu.py
====================
IFG-26 Phase 4B — nnPU Training (Kiryo 2017).

Trains ECFP4-MLP models with non-negative PU risk across a π grid.
Primary evaluation metric: PU-Recall@k and Lift@k (NOT AUROC).

Models:
    nnPU-L0  : ECFP4 (2048) → MLP
    nnPU-LP0 : ECFP4 + protein one-hot (2×(n_proteins+1)) → MLP

π grid: {0.01, 0.02, 0.05, 0.10}

Outputs:
    results/models/nnpu_{model}_pi{pi}.pt   (8 model files)
    results/tables/phase4B_nnpu_metrics.csv
    results/figures/phase4B_pu_recall_curves.png
    docs/phase4B_nnpu_report.md
"""

import os
import sys

# --- Environment Guards (WinError 127 / OMP Conflict Fix) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONNOUSERSITE"] = "1"

if sys.platform == "win32":
    # 1. Strict Version Check
    # prevents WinError 127 caused by running 3.11 torch with 3.13 python
    if sys.version_info[:2] != (3, 11):
        print("\n" + "!" * 70)
        print(f"CRITICAL ERROR: WRONG PYTHON VERSION DETECTED ({sys.version.split()[0]})")
        print("This script MUST be run with Python 3.11 from the 'ifg26' environment.")
        print("Please run using the full path:")
        print(r'  & "C:\Users\Aadi Nair\miniconda3\miniconda4\envs\ifg26\python.exe" scripts/phase4_train_nnpu.py')
        print("!" * 70 + "\n")
        sys.exit(1)

    # 2. Clean PATH of other Miniconda installations
    current_path = os.environ.get("PATH", "").split(os.pathsep)
    clean_path = [p for p in current_path if "miniconda3" not in p.lower() or "ifg26" in p.lower()]
    os.environ["PATH"] = os.pathsep.join(clean_path)

    # 3. Force add ONLY the torch DLL directory (prioritizing torch's internal deps)
    env_base = r"C:\Users\Aadi Nair\miniconda3\miniconda4\envs\ifg26"
    torch_lib = os.path.join(env_base, "Lib", "site-packages", "torch", "lib")
    
    if os.path.exists(torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
        # Put torch_lib at the absolute front of PATH
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ifg26.evaluation.pu_metrics import pu_eval_report, predictive_entropy

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase4_default.yaml"


# ---------------------------------------------------------------------------
# Logging & utilities
# ---------------------------------------------------------------------------
class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def setup_logging(name="phase4_nnpu"):
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt); lg.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); lg.addHandler(sh)
    return lg


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int],
                 dropout: float, batch_norm: bool):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# nnPU loss (Kiryo et al. 2017)
# ---------------------------------------------------------------------------
def nnpu_loss(pos_out: torch.Tensor, unl_out: torch.Tensor,
              pi: float) -> tuple[torch.Tensor, dict]:
    """
    Non-negative PU risk estimator.

    R_pu  = π·R+(P) + max(0, R-(U) - π·R-(P))
    where R+(·) = E[loss(f(x))]     over labeled P
          R-(·) = E[loss(-f(x))]   over labeled P or unlabeled U

    Loss used: sigmoid cross-entropy gives smooth gradients.
    """
    pos_loss = nn.BCELoss(reduction="none")
    neg_loss = nn.BCELoss(reduction="none")

    ones  = torch.ones_like(pos_out)
    zeros = torch.zeros_like(pos_out)
    u_zeros = torch.zeros_like(unl_out)

    r_p_pos = pos_loss(pos_out, ones).mean()        # E[l(f(x)) | P]
    r_p_neg = neg_loss(pos_out, zeros).mean()       # E[l(-f(x)) | P]  [via BCE with 0]
    r_u_neg = neg_loss(unl_out, u_zeros).mean()     # E[l(-f(x)) | U]

    neg_risk = r_u_neg - pi * r_p_neg               # non-negative clamp
    nnpu = pi * r_p_pos + torch.clamp(neg_risk, min=0.0)

    info = {
        "r_p_pos": float(r_p_pos),
        "r_p_neg": float(r_p_neg),
        "r_u_neg": float(r_u_neg),
        "neg_risk_raw": float(neg_risk),
        "nnpu_loss": float(nnpu),
    }
    return nnpu, info


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_features(cfg: dict) -> dict:
    fd = ROOT / cfg["inputs"]["features_dir"] if "features_dir" in cfg.get("inputs", {}) \
         else ROOT / cfg["outputs"]["features_dir"]
    feats_dir = ROOT / "data" / "features"
    pu_dir    = ROOT / cfg["outputs"]["pu_dir"]

    ecfp_mat = np.load(str(feats_dir / "ligands_ecfp4.npy")).astype(np.float32)
    lig_idx  = pd.read_parquet(feats_dir / "ligand_index.parquet")
    p_idx    = pd.read_parquet(feats_dir / "protein_index.parquet")
    bins_df  = pd.read_parquet(feats_dir / "test_similarity_bins.parquet")
    pool_P   = pd.read_parquet(pu_dir / "pool_P_scaffold.parquet")
    pool_U   = pd.read_parquet(pu_dir / "pool_U_scaffold.parquet")

    return {
        "ecfp": ecfp_mat,
        "lig_idx": lig_idx,
        "p_idx": p_idx,
        "bins_df": bins_df,
        "pool_P": pool_P,
        "pool_U": pool_U,
        "n_proteins": len(p_idx),
    }


def make_ecfp_tensor(df: pd.DataFrame, ecfp_mat: np.ndarray) -> torch.Tensor:
    rows = df["ligand_feature_row"].clip(0).values
    return torch.from_numpy(ecfp_mat[rows])


def make_protein_onehot(df: pd.DataFrame, n_proteins: int) -> torch.Tensor:
    n_p1 = n_proteins + 1
    e3_oh  = np.eye(n_p1, dtype=np.float32)[
        df["e3_label_id"].clip(-1).values + 1]
    tgt_oh = np.eye(n_p1, dtype=np.float32)[
        df["target_label_id"].clip(-1).values + 1]
    return torch.from_numpy(np.hstack([e3_oh, tgt_oh]))


def make_features(df: pd.DataFrame, model_type: str,
                  ecfp_mat: np.ndarray, n_proteins: int) -> torch.Tensor:
    ecfp = make_ecfp_tensor(df, ecfp_mat)
    if model_type == "L0":
        return ecfp
    elif model_type == "LP0":
        prot = make_protein_onehot(df, n_proteins)
        return torch.cat([ecfp, prot], dim=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module, optimizer: optim.Optimizer,
    p_loader: DataLoader, u_loader: DataLoader,
    val_p_X: torch.Tensor, pi: float,
    cfg: dict, lg: logging.Logger,
    model_tag: str,
) -> tuple[nn.Module, list[float], list[float]]:
    train_cfg = cfg.get("nnpu", {}).get("training", {})
    epochs  = train_cfg.get("epochs", 100)
    patience = train_cfg.get("patience", 15)

    train_losses, val_scores = [], []
    best_val = -1; best_state = None; no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0; n_batch = 0

        u_iter = iter(u_loader)
        for (p_X,) in p_loader:
            try:
                (u_X,) = next(u_iter)
            except StopIteration:
                u_iter = iter(u_loader)
                (u_X,) = next(u_iter)

            optimizer.zero_grad()
            pos_out = model(p_X)
            unl_out = model(u_X)
            loss, _ = nnpu_loss(pos_out, unl_out, pi)
            loss.backward()
            # Gradient clipping per Kiryo 2017
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batch += 1

        avg_loss = epoch_loss / max(n_batch, 1)
        train_losses.append(avg_loss)

        # Val: mean score on val positives
        model.eval()
        with torch.no_grad():
            val_sc = model(val_p_X).mean().item()
        val_scores.append(val_sc)

        if epoch % 10 == 0:
            lg.info(f"    [{model_tag} π={pi}] epoch {epoch:3d} | "
                    f"loss={avg_loss:.4f} | val_pos_mean={val_sc:.4f}")

        # Early stopping
        if val_sc > best_val:
            best_val = val_sc; best_state = {k: v.clone()
                for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                lg.info(f"    [{model_tag}] Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, train_losses, val_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 4B — nnPU Training  {ts}")
    lg.info("=" * 70)

    torch.manual_seed(cfg["nnpu"]["training"]["seed"])
    np.random.seed(cfg["nnpu"]["training"]["seed"])

    # Load all features
    feats = load_features(cfg)
    ecfp  = feats["ecfp"]
    pool_P = feats["pool_P"]; pool_U = feats["pool_U"]
    bins_df = feats["bins_df"]; n_prot = feats["n_proteins"]

    lg.info(f"  Pool P: {len(pool_P)} | Pool U: {len(pool_U)}")
    lg.info(f"  Similarity bins loaded: {len(bins_df)} test ligands")

    train_cfg = cfg["nnpu"]["training"]
    mlp_cfg   = cfg["nnpu"]["mlp"]
    pi_grid   = cfg["nnpu"]["pi_grid"]
    k_list    = cfg["nnpu"]["recall_k"]
    bs        = train_cfg["batch_size"]

    all_metrics: list[dict] = []
    records:     list[dict] = []

    models_dir  = ROOT / cfg["outputs"]["models_dir"]
    tables_dir  = ROOT / cfg["outputs"]["tables_dir"]
    figures_dir = ROOT / cfg["outputs"]["figures_dir"]
    for d in [models_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Recall curves for plotting
    fig_rc, ax_rc = plt.subplots(figsize=(10, 6))

    for model_type in cfg["nnpu"]["models"]:
        lg.info(f"\n{'='*50}")
        lg.info(f"Model type: {model_type}")

        # ── Build train/val/test feature tensors ─────────────────────
        p_train = pool_P[pool_P["split"] == "train"]
        p_val   = pool_P[pool_P["split"] == "val"]
        p_test  = pool_P[pool_P["split"] == "test"]

        X_p_train = make_features(p_train, model_type, ecfp, n_prot)
        X_p_val   = make_features(p_val,   model_type, ecfp, n_prot)
        X_p_test  = make_features(p_test,  model_type, ecfp, n_prot)

        # U uses all splits (unlabeled)
        X_u_all   = make_features(pool_U, model_type, ecfp, n_prot)
        # U train subset for loss computation
        u_train   = pool_U[pool_U["source_split"] == "train"]
        X_u_train = make_features(u_train, model_type, ecfp, n_prot) \
                    if len(u_train) > 0 else X_u_all[:min(len(X_u_all), len(X_p_train))]

        in_dim = X_p_train.shape[1]
        lg.info(f"  Input dim: {in_dim} | P_train: {len(X_p_train)} | "
                f"U_train: {len(X_u_train)} | P_test: {len(X_p_test)}")

        p_loader = DataLoader(TensorDataset(X_p_train), batch_size=bs, shuffle=True)
        u_loader = DataLoader(TensorDataset(X_u_train), batch_size=bs, shuffle=True)

        for pi in pi_grid:
            model_tag = f"nnpu_{model_type}_pi{pi}"
            lg.info(f"\n  Training {model_tag} ...")

            # Fresh model for each (type, π)
            model = MLP(
                in_dim=in_dim,
                hidden_dims=mlp_cfg.get("hidden_dims", [512, 128]),
                dropout=mlp_cfg.get("dropout", 0.3),
                batch_norm=mlp_cfg.get("batch_norm", True),
            )
            optimizer = optim.Adam(
                model.parameters(),
                lr=train_cfg["lr"],
                weight_decay=train_cfg.get("weight_decay", 1e-4),
            )

            model, train_losses, val_scores = train_model(
                model, optimizer, p_loader, u_loader,
                X_p_val, pi, cfg, lg, model_tag,
            )

            # Save checkpoint
            ckpt_path = models_dir / f"{model_tag}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "model_type": model_type, "pi": pi,
                "in_dim": in_dim, "mlp_cfg": mlp_cfg,
                "train_losses": train_losses,
                "val_scores": val_scores,
                "timestamp": ts,
                "seed": train_cfg["seed"],
            }, ckpt_path)
            lg.info(f"  Saved: {ckpt_path.name}")

            # ── Evaluation ──────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                p_test_scores = model(X_p_test).numpy()
                u_all_scores  = model(X_u_all).numpy()

            test_iks = p_test["ligand_inchikey"].tolist()
            report   = pu_eval_report(
                model_name=model_tag, pi=pi,
                p_test_scores=p_test_scores,
                u_scores=u_all_scores,
                p_test_inchikeys=test_iks,
                bins_df=bins_df,
                k_list=k_list,
            )
            all_metrics.append(report)

            # Flatten into rows for CSV
            for bin_key, bin_data in report["metrics"].items():
                if isinstance(bin_data, dict) and "skipped" not in bin_data:
                    for k_key, k_data in bin_data.items():
                        if isinstance(k_data, dict):
                            records.append({
                                "model": model_type,
                                "pi": pi,
                                "bin": bin_key,
                                "k": k_key,
                                "recall": k_data.get("recall"),
                                "lift":   k_data.get("lift"),
                                "n_P":    k_data.get("n_P"),
                                "n_U":    k_data.get("n_U"),
                            })

            # Add to recall curve plot (k=5%, overall)
            k5_overall = report["metrics"].get("overall", {}).get("k=0.05", {})
            ax_rc.scatter([pi], [k5_overall.get("recall", 0)],
                          label=f"{model_tag} recall@5%", s=80)

            lg.info(f"  [{model_tag}] Overall Recall@5%: "
                    f"{k5_overall.get('recall','?')} | "
                    f"Lift@5%: {k5_overall.get('lift','?')}")
            excl_e = report["metrics"].get("excl_bin_E", {}).get("k=0.05", {})
            lg.info(f"  [{model_tag}] Excl Bin-E Recall@5%: "
                    f"{excl_e.get('recall','?')} (Policy 2 sensitivity)")

    # ── Save metrics table ────────────────────────────────────────────
    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(tables_dir / "phase4B_nnpu_metrics.csv", index=False)
    lg.info(f"\n  Written: phase4B_nnpu_metrics.csv ({len(metrics_df)} rows)")

    # ── Save metrics JSON ─────────────────────────────────────────────
    with open(tables_dir / "phase4B_nnpu_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, cls=_NpEncoder)

    # ── Recall curve plot ─────────────────────────────────────────────
    ax_rc.set_xlabel("π (prior P(Y=1))", fontsize=11)
    ax_rc.set_ylabel("PU-Recall@5%", fontsize=11)
    ax_rc.set_title("IFG-26 nnPU: PU-Recall@5% vs π (test set)", fontsize=12)
    ax_rc.legend(fontsize=8, ncol=2)
    ax_rc.set_xscale("log")
    fig_rc.tight_layout()
    fig_rc.savefig(figures_dir / "phase4B_pu_recall_curves.png", dpi=150)
    plt.close(fig_rc)

    # ── Report ────────────────────────────────────────────────────────
    # Find best model (highest excl-Bin-E Recall@5%)
    best_rows = [(r["model"], r["pi"],
                  r["metrics"].get("excl_bin_E", {}).get("k=0.05", {}).get("recall", 0))
                 for r in all_metrics]
    best_model, best_pi, best_recall = max(best_rows, key=lambda x: x[2])

    report_md = f"""# IFG-26 Phase 4B — nnPU Training Report

_Generated: {ts}_

## Headline Metrics (Primary: PU-Recall@k, Lift@k)

> Per editorial policy: AUROC is NOT reported for PU-setting headline results.
> All results shown both overall and excluding Bin E (NN>0.95) per Policy 2.

## Best Configuration

| Item | Value |
|---|---|
| Best model | {best_model} |
| Best π | {best_pi} |
| Excl. Bin-E Recall@5% | {best_recall:.4f} |

## All Results

See `results/tables/phase4B_nnpu_metrics.csv` for full breakdown.

### Key Columns:
- `bin`: overall / excl_bin_E / bin_A / bin_B / bin_C / bin_D / bin_E
- `recall`: PU-Recall@k = fraction of test positives in top-k% of U∪P
- `lift`: Lift@k = Recall@k / (|P|/|P∪U|) — 1.0 = random

## π Sensitivity

The π grid `{pi_grid}` spans the plausible prior range for molecular glues.
Full sensitivity table is in the CSV. The recall curves plot is at:
`results/figures/phase4/phase4B_pu_recall_curves.png`

## Training Details

| Parameter | Value |
|---|---|
| Epochs | {cfg['nnpu']['training']['epochs']} (early stop patience {cfg['nnpu']['training']['patience']}) |
| Batch size | {cfg['nnpu']['training']['batch_size']} |
| LR | {cfg['nnpu']['training']['lr']} |
| MLP hidden | {cfg['nnpu']['mlp']['hidden_dims']} |
| Dropout | {cfg['nnpu']['mlp']['dropout']} |
| Loss | nnPU non-negative risk (Kiryo 2017) |
| Seed | {cfg['nnpu']['training']['seed']} |

## Protein Shortcut Check (to be filled by Phase 4D)

Compare Bin-A Recall@5%:
- L0 (ligand-only): see metrics CSV, `bin=bin_A k=0.05 model=L0`
- LP0 (lig+protein): see metrics CSV, `bin=bin_A k=0.05 model=LP0`

If LP0_BinA - L0_BinA > 0.05: see `docs/phase4D_protein_shortcut_diagnosis.md`.
"""

    with open(ROOT / "docs" / "phase4B_nnpu_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    lg.info("  Written: docs/phase4B_nnpu_report.md")

    lg.info("\n" + "=" * 70)
    lg.info("IFG-26 Phase 4B nnPU Training — COMPLETE")
    lg.info(f"  Best model: {best_model} π={best_pi} Recall@5%={best_recall:.4f}")
    lg.info(f"  Models saved: {len(cfg['nnpu']['pi_grid']) * len(cfg['nnpu']['models'])} .pt files")
    lg.info("=" * 70)
    lg.info("\nNext: python scripts/phase4_generate_pmd_negatives.py")


if __name__ == "__main__":
    main()
