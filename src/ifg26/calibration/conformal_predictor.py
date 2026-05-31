"""
conformal_predictor.py
======================
Genuine Split Conformal Inference Predictor for target-conditioned ML.
Calculates mathematically rigorous Lower Confidence Bounds (LCB) for out-of-sample rankings.
"""

import numpy as np

class SplitConformalRanker:
    def __init__(self, alpha: float = 0.1):
        """
        Parameters:
            alpha: Significance level (e.g. 0.1 for 90% confidence coverage).
        """
        self.alpha = alpha
        self.nonconformity_scores = None
        
    def fit(self, y_cal: np.ndarray, prob_cal: np.ndarray):
        """
        Fits the conformal predictor using a calibration set.
        
        For binary classification, the non-conformity score is defined as:
            For true positive binders (y = 1): 1.0 - P(y=1|X)
            For true negative decoys (y = 0): P(y=1|X)
        """
        y_cal = np.array(y_cal)
        prob_cal = np.array(prob_cal)
        
        if len(y_cal) == 0 or len(prob_cal) == 0:
            raise ValueError("Calibration data cannot be empty.")
            
        # Non-conformity score: distance between label and prediction
        self.nonconformity_scores = np.where(y_cal == 1, 1.0 - prob_cal, prob_cal)
        
    def predict_lcb(self, prob_new: np.ndarray) -> np.ndarray:
        """
        Computes the Lower Confidence Bound (LCB) for new candidates.
        
        LCB_i = prob_new_i - q_alpha
        where q_alpha is the (1 - alpha)(1 + 1/n) empirical quantile of 
        non-conformity scores on the calibration set.
        """
        if self.nonconformity_scores is None:
            raise ValueError("Conformal Predictor must be fitted before prediction.")
            
        prob_new = np.array(prob_new)
        n = len(self.nonconformity_scores)
        
        # Calculate (1 - alpha)(1 + 1/n) quantile
        quantile_pct = 100.0 * (1.0 - self.alpha) * (1.0 + 1.0 / n)
        quantile_pct = np.clip(quantile_pct, 0.0, 100.0)
        
        q_alpha = np.percentile(self.nonconformity_scores, quantile_pct)
        
        # Conformal LCB
        lcb = prob_new - q_alpha
        return np.clip(lcb, 0.0, 1.0)
        
    def compute_p_values(self, prob_new: np.ndarray) -> np.ndarray:
        """
        Computes conformal p-values for new candidates representing the null hypothesis 
        that the compound is a non-binder (decoy control).
        
        p_val = (sum(nonconformity_scores >= nonconformity_new) + 1) / (n + 1)
        """
        if self.nonconformity_scores is None:
            raise ValueError("Conformal Predictor must be fitted before prediction.")
            
        prob_new = np.array(prob_new)
        n = len(self.nonconformity_scores)
        
        p_vals = []
        for p in prob_new:
            # Under null hypothesis y = 0, non-conformity is p
            nc_new = p
            count_greater = np.sum(self.nonconformity_scores >= nc_new)
            p_vals.append((count_greater + 1.0) / (n + 1.0))
            
        return np.array(p_vals)

if __name__ == "__main__":
    print("Testing Conformal Predictor...")
    # Mock calibration
    y_cal = np.array([1, 1, 1, 0, 0, 0])
    prob_cal = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.15])
    
    ranker = SplitConformalRanker(alpha=0.1)
    ranker.fit(y_cal, prob_cal)
    
    # New predictions
    prob_new = np.array([0.95, 0.5, 0.1])
    lcbs = ranker.predict_lcb(prob_new)
    p_vals = ranker.compute_p_values(prob_new)
    
    print("LCBs:    ", lcbs)
    print("P-values:", p_vals)
    assert lcbs[0] > lcbs[1] >= lcbs[2]
    print("Passed!")
