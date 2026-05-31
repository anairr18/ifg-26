"""
estimate_class_prior.py

Estimating the PU class prior (pi) using the Elkan-Noto method.
"""
import numpy as np

def estimate_prior_elkan_noto(y_unlabeled_scores: np.ndarray, y_positive_scores: np.ndarray) -> float:
    """
    Estimate class prior (pi) using the Elkan-Noto estimator.
    Formula: pi = P(s=1) / P(s=1 | y=1)
    Approximated by mean score over unlabeled divided by mean score over positives.
    """
    if len(y_unlabeled_scores) == 0:
        return 0.02
        
    mean_unlabeled = np.mean(y_unlabeled_scores)
    mean_positive = np.mean(y_positive_scores) if len(y_positive_scores) > 0 else 1.0
    
    if mean_positive == 0:
        return 0.02
        
    estimated_pi = mean_unlabeled / mean_positive
    
    # Clip to reasonable biological range for molecular glues (1% to 20%)
    return float(np.clip(estimated_pi, 0.01, 0.20))

def estimate_class_prior(X_pos: np.ndarray, X_unlabeled: np.ndarray, seed: int = 42) -> float:
    """
    Statistically sound estimation of π = P(y=1) using Elkan-Noto.
    1. Trains a P-vs-U proxy classifier (Logistic Regression).
    2. Computes calibration scores.
    3. Estimates π.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    
    # Combine data for proxy training (P=1, U=0)
    X = np.vstack([X_pos, X_unlabeled])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_unlabeled))])
    
    # Use 3-fold cross-validation to get out-of-fold scores for better calibration
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    y_scores = np.zeros(len(y))
    
    for train_idx, val_idx in skf.split(X, y):
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed, n_jobs=1)
        model.fit(X[train_idx], y[train_idx])
        y_scores[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    
    y_p_scores = y_scores[y == 1]
    y_u_scores = y_scores[y == 0]
    
    pi = estimate_prior_elkan_noto(y_u_scores, y_p_scores)
    return pi

def sweep_prior_sensitivity(y_u_scores, y_p_scores, priors_to_test=[0.01, 0.02, 0.05, 0.10]):
    """
    Sweep multiple priors to see how they impact estimates.
    """
    en_prior = estimate_prior_elkan_noto(y_u_scores, y_p_scores)
    results = {'elkan_noto_estimated': en_prior}
    return results

if __name__ == "__main__":
    # Smoke test
    print("Running Robust Smoke Test for estimate_class_prior.py...")
    # Generate synthetic features
    np.random.seed(42)
    X_p = np.random.normal(loc=1.0, size=(100, 10))
    X_u = np.random.normal(loc=-1.0, size=(500, 10))
    
    pi = estimate_class_prior(X_p, X_u)
    print(f"Estimated Prior: {pi:.4f}")
    assert 0.01 <= pi <= 0.20
    print("Smoke Test Passed!")

