import numpy as np
from ifg26.models.pu_learning.estimate_class_prior import estimate_class_prior

def test_pi_estimation_deterministic():
    """Verify that estimation is deterministic given a fixed seed."""
    np.random.seed(42)
    X_pos = np.random.normal(loc=1.0, size=(100, 10))
    X_unl = np.random.normal(loc=-1.0, size=(500, 10))
    
    pi1 = estimate_class_prior(X_pos, X_unl, seed=42)
    pi2 = estimate_class_prior(X_pos, X_unl, seed=42)
    
    assert pi1 == pi2
    assert 0.01 <= pi1 <= 0.20

def test_pi_stability_across_splits():
    """Verify pi stability when data is shuffled/split."""
    np.random.seed(42)
    X_all_pos = np.random.normal(loc=1.0, size=(200, 10))
    X_all_unl = np.random.normal(loc=-1.0, size=(1000, 10))
    
    # Split 1
    idx_p1 = np.random.choice(200, 100, replace=False)
    idx_u1 = np.random.choice(1000, 500, replace=False)
    pi_split1 = estimate_class_prior(X_all_pos[idx_p1], X_all_unl[idx_u1], seed=1)
    
    # Split 2 (different subset)
    idx_p2 = np.random.choice(200, 100, replace=False)
    idx_u2 = np.random.choice(1000, 500, replace=False)
    pi_split2 = estimate_class_prior(X_all_pos[idx_p2], X_all_unl[idx_u2], seed=2)
    
    print(f"pi Split 1: {pi_split1:.4f}")
    print(f"pi Split 2: {pi_split2:.4f}")
    
    # Stability: should be within a reasonable threshold (e.g. 0.05 absolute)
    assert abs(pi_split1 - pi_split2) < 0.05

if __name__ == "__main__":
    print("Running pi Estimation Stability Validation...")
    test_pi_estimation_deterministic()
    test_pi_stability_across_splits()
    print("All validation checks passed.")
