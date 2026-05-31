"""
statistical_validation.py

Run bootstrap resampling to build confidence intervals for evaluation metrics.
"""
import numpy as np

def run_bootstrap_metrics(y_true: np.ndarray, y_scores: np.ndarray, metric_funcs: dict, iterations: int = 100, random_state: int = 42) -> dict:
    """
    Run bootstrap iterations over predictions to determine variance using vectorized numpy sampling.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    results = {name: [] for name in metric_funcs.keys()}
    
    # Generate bootstrap indices once
    bootstrap_indices = rng.choice(n_samples, size=(iterations, n_samples), replace=True)
    
    for i in range(iterations):
        idx = bootstrap_indices[i]
        y_t = y_true[idx]
        y_s = y_scores[idx]
        for name, func in metric_funcs.items():
            results[name].append(func(y_t, y_s))
            
    summary = {}
    for name, vals in results.items():
        clean_vals = [v for v in vals if not np.isnan(v)]
        if not clean_vals:
            summary[name] = {
                'metric_name': name,
                'mean': float('nan'), 
                'ci_lower': float('nan'), 
                'ci_upper': float('nan'),
                'bootstrap_iterations': iterations
            }
        else:
            summary[name] = {
                'metric_name': name,
                'mean': float(np.mean(clean_vals)),
                'ci_lower': float(np.percentile(clean_vals, 2.5)),
                'ci_upper': float(np.percentile(clean_vals, 97.5)),
                'bootstrap_iterations': iterations
            }
    return summary

if __name__ == "__main__":
    print("Running Smoke Test for statistical_validation.py...")
    y_t = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    y_s = np.array([0.9, 0.1, 0.8, 0.4, 0.7, 0.2, 0.3, 0.1, 0.6, 0.4])
    
    def dummy_metric(yt, ys):
        return np.mean((ys > 0.5) == yt)
        
    res = run_bootstrap_metrics(y_t, y_s, {'dummy_acc': dummy_metric}, iterations=10)
    assert 'dummy_acc' in res
    assert 'ci_lower' in res['dummy_acc']
    print("Smoke Test Passed!")

