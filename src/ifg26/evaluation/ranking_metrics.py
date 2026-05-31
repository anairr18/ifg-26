"""
ranking_metrics.py

Ranking and enrichment evaluation metrics (Recall@K, Lift@K).
Defaults to wrapping custom pu_metrics if available.
"""
import numpy as np

def recall_at_k(y_true, y_scores, k_percent):
    """
    Compute Recall at top K percent of predictions.
    
    Args:
        y_true: Ground truth labels (1 or 0)
        y_scores: Predicted probabilities
        k_percent: Percentage to consider (e.g., 5 for top 5%)
    """
    n_samples = len(y_true)
    k = int((k_percent / 100.0) * n_samples)
    if k == 0:
        return 0.0
        
    # Get indices of top K predictions
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    
    true_positives_in_top_k = np.sum(y_true[top_k_indices])
    total_positives = np.sum(y_true)
    
    if total_positives == 0:
        return float('nan')
        
    return true_positives_in_top_k / total_positives

def lift_at_k(y_true, y_scores, k_percent):
    """
    Compute Enrichment Factor (Lift) at top K percent.
    Lift = (TP in top k / k) / (Total P / Total N)
    """
    n_samples = len(y_true)
    k = int((k_percent / 100.0) * n_samples)
    if k == 0:
        return 0.0
        
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    
    true_positives_in_top_k = np.sum(y_true[top_k_indices])
    total_positives = np.sum(y_true)
    
    if total_positives == 0:
        return float('nan')
        
    precision_at_k = true_positives_in_top_k / k
    base_rate = total_positives / n_samples
    
    return precision_at_k / base_rate

def compute_ndcg(y_true, y_scores, k=None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.
    """
    from sklearn.metrics import ndcg_score
    # Reshape for single sample ranking
    yt = np.asarray(y_true).reshape(1, -1)
    ys = np.asarray(y_scores).reshape(1, -1)
    return float(ndcg_score(yt, ys, k=k))

if __name__ == "__main__":
    print("Smoke Test for ranking_metrics.py")
    r = recall_at_k(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]), 50)
    l = lift_at_k(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]), 50)
    assert r == 1.0 and l == 2.0
    print("Passed!")
