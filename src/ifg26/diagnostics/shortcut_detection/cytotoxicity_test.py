"""
cytotoxicity_test.py

Test if model simply learns that a compound is broadly cytotoxic
instead of being a specific molecular glue (Shortcut Mechanism).
"""
import numpy as np
import warnings

def check_cytotoxicity_bias(y_scores: np.ndarray, smiles_list: list) -> dict:
    """
    Correlation between model predictions and ChEMBL broad cytotoxicity assays.
    """
    return {
        "diagnostic_name": "cytotoxicity_spearman",
        "metric_value": 0.0,
        "diagnostic_status": "SKIPPED",
        "implementation_status": "requires_labels"
    }

if __name__ == "__main__":
    print("Smoke Test for cytotoxicity_test.py")
    res = check_cytotoxicity_bias(np.array([0.9, 0.1]), ["CC", "C"])
    assert res["diagnostic_status"] == "SKIPPED"
    print("Passed!")
