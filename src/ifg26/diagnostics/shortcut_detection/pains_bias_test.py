"""
pains_bias_test.py

Test if the model is biased identifying Pan Assay Interference Compounds (PAINS)
as active glues falsely.
"""

def _check_rdkit_safe():
    import subprocess, sys
    try:
        res = subprocess.run([sys.executable, "-c", "import rdkit"], capture_output=True)
        return res.returncode == 0
    except Exception:
        return False

def check_pains_bias(smiles_list: list, y_scores: list) -> dict:
    """
    Evaluate if high-scoring predictions are enriched for PAINS alerts using RDKit FilterCatalog.
    """
    diagnostic_name = "pains_enrichment"
    if not _check_rdkit_safe():
        return {
            "diagnostic_name": diagnostic_name,
            "metric_value": 0.0,
            "diagnostic_status": "SKIPPED",
            "implementation_status": "requires_rdkit_clean_install"
        }
        
    try:
        from rdkit import Chem
        from rdkit.Chem import FilterCatalog
        
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        
        pains_count = 0
        pains_scores = []
        non_pains_scores = []
        
        for smi, score in zip(smiles_list, y_scores):
            mol = Chem.MolFromSmiles(smi)
            if mol and catalog.HasMatch(mol):
                pains_count += 1
                pains_scores.append(score)
            else:
                non_pains_scores.append(score)
                
        if pains_count == 0 or len(non_pains_scores) == 0:
            enrichment = 1.0 # Baseline if no pains
        else:
            enrichment = (sum(pains_scores) / len(pains_scores)) / (sum(non_pains_scores) / len(non_pains_scores))
            
        status = "PASS" if enrichment < 1.2 else "WARNING" if enrichment < 1.5 else "FAIL"
        
        return {
            "diagnostic_name": diagnostic_name,
            "metric_value": enrichment,
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

if __name__ == "__main__":
    print("Smoke Test for pains_bias_test.py")
    res = check_pains_bias(["C", "CC"], [0.9, 0.1])
    assert res["diagnostic_name"] == "pains_enrichment"
    print("Passed!")
