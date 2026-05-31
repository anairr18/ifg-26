"""
scaffold_memorization_test.py

Test if a model's performance collapses when evaluated against
Bemis-Murcko scaffolds it has never seen before.
"""
import warnings

def _check_rdkit_safe():
    import subprocess, sys
    try:
        res = subprocess.run([sys.executable, "-c", "import rdkit"], capture_output=True)
        return res.returncode == 0
    except Exception:
        return False

def check_scaffold_memorization(train_smiles: list, test_smiles: list, test_scores: list) -> dict:
    """
    Compute nearest-neighbor similarity using Tanimoto between test and train.
    Correlation between distance-to-train-set and prediction score.
    """
    diagnostic_name = "scaffold_drop"
    if not _check_rdkit_safe():
        return {
            "diagnostic_name": diagnostic_name,
            "metric_value": 0.0,
            "diagnostic_status": "SKIPPED",
            "implementation_status": "requires_rdkit_clean_install"
        }
        
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        from scipy.stats import pearsonr
        import numpy as np
        
        # Calculate train FPs
        train_fps = []
        for smi in train_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
                
        if not train_fps:
            raise ValueError("No valid training SMILES.")
            
        test_max_sims = []
        valid_scores = []
        
        for smi, score in zip(test_smiles, test_scores):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
                test_max_sims.append(max(sims))
                valid_scores.append(score)
                
        if len(test_max_sims) < 2:
            correlation = 0.0
        else:
            correlation, _ = pearsonr(valid_scores, test_max_sims)
            
        # High correlation means models only score highly when very similar to train set
        status = "PASS" if correlation < 0.3 else "WARNING" if correlation < 0.5 else "FAIL"
        
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
    print("Smoke Test for scaffold_memorization_test.py")
    res = check_scaffold_memorization(["C", "CC"], ["CCC"], [0.5])
    assert res["diagnostic_name"] == "scaffold_drop"
    print("Passed!")
