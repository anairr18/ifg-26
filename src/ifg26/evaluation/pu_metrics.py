"""
pu_metrics.py
=============
IFG-26 — PU Evaluation Metrics.

Implements PU-Recall@k, Lift@k, and per-bin reporting for nnPU models.
These are the PRIMARY headline metrics (NOT AUROC) for the PU learning setting.

Usage:
    from ifg26.evaluation.pu_metrics import pu_eval_report
    report = pu_eval_report(u_scores, p_scores, bins_df, k_list, pi)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


__all__ = ["pu_recall_at_k", "lift_at_k", "pu_eval_report",
           "bin_stratified_metrics", "predictive_entropy"]


def pu_recall_at_k(p_scores: np.ndarray, u_scores: np.ndarray,
                   k_frac: float) -> float:
    """
    PU-Recall@k: fraction of positives (P) ranked in the top-k% of ALL scored items
    (P ∪ U), where k is expressed as a fraction of |P ∪ U|.

    This is the primary headline metric. It answers:
    "If we flag the top k% of candidates, what fraction of real positives do we catch?"

    Args:
        p_scores: 1D array of scores for labeled positives
        u_scores: 1D array of scores for unlabeled candidates
        k_frac:   fraction of |P ∪ U| to consider (e.g. 0.05 for top 5%)

    Returns:
        Recall = |P in top-k| / |P|
    """
    all_scores = np.concatenate([p_scores, u_scores])
    n_total = len(all_scores)
    k = max(1, int(np.ceil(k_frac * n_total)))
    threshold = np.partition(all_scores, -k)[-k]
    n_p_in_topk = (p_scores >= threshold).sum()
    return float(n_p_in_topk) / max(1, len(p_scores))


def lift_at_k(p_scores: np.ndarray, u_scores: np.ndarray,
              k_frac: float) -> float:
    """
    Lift@k: Recall@k / baseline_recall
    where baseline_recall = |P| / |P ∪ U| (random ranking).
    """
    all_scores = np.concatenate([p_scores, u_scores])
    n_total = len(all_scores)
    baseline = len(p_scores) / n_total  # random rate
    recall = pu_recall_at_k(p_scores, u_scores, k_frac)
    return recall / max(baseline, 1e-9)


def pu_ndcg_at_k(p_scores: np.ndarray, u_scores: np.ndarray, k_frac: float = 0.05) -> float:
    """
    Normalized Discounted Cumulative Gain in PU setting.
    Assumes all items in P are 1, all in U are 0.
    """
    all_scores = np.concatenate([p_scores, u_scores])
    y_true = np.concatenate([np.ones(len(p_scores)), np.zeros(len(u_scores))])
    
    # Reshape for sklearn [n_samples, n_classes]
    yt = y_true.reshape(1, -1)
    ys = all_scores.reshape(1, -1)
    
    n_total = len(all_scores)
    k = max(1, int(np.ceil(k_frac * n_total)))
    
    return float(ndcg_score(yt, ys, k=k))


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Predictive entropy from calibrated probabilities.
    H = -p*log(p) - (1-p)*log(1-p)
    """
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def bin_stratified_metrics(
    p_scores: np.ndarray,
    u_scores: np.ndarray,
    p_inchikeys: list[str],
    bins_df: pd.DataFrame,
    k_list: list[float],
    bin_labels: list[str] = None,
) -> dict:
    """
    Compute PU-Recall@k and Lift@k stratified by similarity bin.
    Only test-set positives have bin assignments available.

    Args:
        p_scores:      scores for test-split positives
        u_scores:      scores for unlabeled pool (ALL splits)
        p_inchikeys:   InChIKeys of test positives (aligned with p_scores)
        bins_df:       DataFrame with compound_inchi_key → similarity_bin
        k_list:        list of k fractions

    Returns:
        dict: bin → {k_frac: {recall, lift}}
    """
    if bin_labels is None:
        bin_labels = ["A", "B", "C", "D", "E"]

    bin_map = bins_df.set_index("compound_inchi_key")["similarity_bin"].to_dict()
    p_bins  = np.array([bin_map.get(ik, "?") for ik in p_inchikeys])

    result = {}

    # Overall (all bins included)
    result["overall"] = {}
    for k in k_list:
        result["overall"][f"k={k}"] = {
            "recall": round(pu_recall_at_k(p_scores, u_scores, k), 4),
            "lift":   round(lift_at_k(p_scores, u_scores, k), 3),
            "ndcg":   round(pu_ndcg_at_k(p_scores, u_scores, k), 4),
            "n_P":    len(p_scores),
            "n_U":    len(u_scores),
        }

    # Exclude Bin E (Policy 2)
    not_e_mask = p_bins != "E"
    if not_e_mask.sum() > 0:
        result["excl_bin_E"] = {}
        for k in k_list:
            result["excl_bin_E"][f"k={k}"] = {
                "recall": round(pu_recall_at_k(p_scores[not_e_mask], u_scores, k), 4),
                "lift":   round(lift_at_k(p_scores[not_e_mask], u_scores, k), 3),
                "ndcg":   round(pu_ndcg_at_k(p_scores[not_e_mask], u_scores, k), 4),
                "n_P":    int(not_e_mask.sum()),
                "n_U":    len(u_scores),
            }

    # Per bin
    for b in bin_labels:
        mask = p_bins == b
        if mask.sum() < 2:
            result[f"bin_{b}"] = {"n_P": int(mask.sum()), "skipped": True}
            continue
        result[f"bin_{b}"] = {}
        for k in k_list:
            result[f"bin_{b}"][f"k={k}"] = {
                "recall": round(pu_recall_at_k(p_scores[mask], u_scores, k), 4),
                "lift":   round(lift_at_k(p_scores[mask], u_scores, k), 3),
                "ndcg":   round(pu_ndcg_at_k(p_scores[mask], u_scores, k), 4),
                "n_P":    int(mask.sum()),
                "n_U":    len(u_scores),
            }

    return result


def pu_eval_report(
    model_name: str,
    pi: float,
    p_test_scores: np.ndarray,
    u_scores: np.ndarray,
    p_test_inchikeys: list[str],
    bins_df: pd.DataFrame,
    k_list: list[float],
) -> dict:
    """Full PU evaluation report for one (model, π) combination."""
    metrics = bin_stratified_metrics(
        p_test_scores, u_scores, p_test_inchikeys, bins_df, k_list
    )
    entropy = predictive_entropy(p_test_scores)
    return {
        "model":    model_name,
        "pi":       pi,
        "metrics":  metrics,
        "p_entropy": {
            "mean":   round(float(entropy.mean()), 4),
            "median": round(float(np.median(entropy)), 4),
            "std":    round(float(entropy.std()),  4),
        },
        "note": (
            "AUROC is NOT reported — PU setting requires PU-Recall@k/Lift@k. "
            "Sensitivity analysis excl. Bin E (NN>0.95) per Policy 2."
        ),
    }
