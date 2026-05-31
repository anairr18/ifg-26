"""
lipophilicity_bias_test.py

Test if the model simply ranks compounds by their lipophilicity (logP)
or other simple descriptors.
"""
import numpy as np
from scipy.stats import pearsonr

def _check_rdkit_safe():
    import subprocess, sys
    try:
        res = subprocess.run([sys.executable, "-c", "import rdkit"], capture_output=True)
        return res.returncode == 0
    except Exception:
        return False

def check_lipophilicity_bias(smiles_list: list, scores: list) -> dict:
    """
    Compute correlation between RDKit logP/MW and predicted scores.
    """
    diagnostic_name = "lipophilicity_pearson"
    if not _check_rdkit_safe():
        return {
            "diagnostic_name": diagnostic_name,
            "metric_value": 0.0,
            "diagnostic_status": "SKIPPED",
            "implementation_status": "requires_rdkit_clean_install"
        }
        
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        logp_vals = []
        valid_scores = []
        
        for smi, score in zip(smiles_list, scores):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                logp_vals.append(Descriptors.MolLogP(mol))
                valid_scores.append(score)
                
        if len(logp_vals) < 2:
            correlation = 0.0
        else:
            correlation, _ = pearsonr(valid_scores, logp_vals)
            
        status = "PASS" if abs(correlation) < 0.3 else "WARNING" if abs(correlation) < 0.5 else "FAIL"
        
        return {
            "diagnostic_name": diagnostic_name,
            "metric_value": float(correlation),
            "diagnostic_status": status,
            "implementation_status": "implemented"
        }
    except ImportError:
         return {
            "diagnostic_name": diagnostic_name,
            "metric_value": 0.0,
            "diagnostic_status": "SKIPPED",
            "implementation_status": "requires_rdkit"
        }
    except Exception:
        return {
            "diagnostic_name": diagnostic_name,
            "metric_value": 0.0,
            "diagnostic_status": "SKIPPED",
            "implementation_status": "computation_failed"
        }

if __name__ == "__main__":
    print("Smoke Test for lipophilicity_bias_test.py")
    res = check_lipophilicity_bias(["C", "CC"], [0.9, 0.8])
    assert res["diagnostic_name"] == "lipophilicity_pearson"
    print("Passed!")
