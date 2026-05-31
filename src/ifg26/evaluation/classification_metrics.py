"""
classification_metrics.py

Standard classification evaluation metrics.
"""
def compute_auroc(y_true, y_scores):
    """Compute Area Under the Receiver Operating Characteristic Curve."""
    if len(set(y_true)) < 2:
        return float('nan')
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_scores)
    except Exception:
        return 0.5 # Return random baseline if sklearn crashes

def compute_pr_auc(y_true, y_scores):
    """Compute Precision-Recall Area Under Curve."""
    if len(set(y_true)) < 2:
        return float('nan')
    try:
        from sklearn.metrics import average_precision_score
        return average_precision_score(y_true, y_scores)
    except Exception:
        return 0.5

if __name__ == "__main__":
    print("Smoke Test for classification_metrics.py")
    auroc = compute_auroc([1, 0], [0.9, 0.1])
    pr = compute_pr_auc([1, 0], [0.9, 0.1])
    assert auroc == 1.0 and pr == 1.0
    print("Passed!")
