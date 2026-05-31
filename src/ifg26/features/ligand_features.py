"""
ligand_features.py

Canonicalization and ECFP4 fingerprint generation with disk caching.
"""
import os
import pickle
import numpy as np

CACHE_FILE = 'cache/ecfp4_cache.pkl'

def get_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    # Ensure directory exists
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def _check_rdkit_safe():
    import subprocess, sys
    try:
        res = subprocess.run([sys.executable, "-c", "import rdkit"], capture_output=True)
        return res.returncode == 0
    except Exception:
        return False

def generate_ecfp4(smiles_list, radius: int = 2, nBits: int = 2048):
    """Generate ECFP4 fingerprints with caching to avoid recomputation."""
    if not _check_rdkit_safe():
        raise ImportError(
            "RDKit is missing or broken (DLL error). Please install RDKit (e.g. `conda install -c conda-forge rdkit`) "
            "to generate ECFP4 fingerprints."
        )

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError(
            "RDKit is missing. Please install RDKit (e.g. `conda install -c conda-forge rdkit` "
            "or `pip install rdkit`) to generate ECFP4 fingerprints."
        )
        
    cache = get_cache()
    features = []
    updated = False
    
    for smi in smiles_list:
        if smi in cache:
            features.append(cache[smi])
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Add zero vector for invalid smiles
                vec = np.zeros(nBits, dtype=np.int8) 
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                vec = np.array(fp)
            
            cache[smi] = vec
            features.append(vec)
            updated = True
            
    if updated:
        save_cache(cache)
        
    return np.array(features)

def generate_graph_features(smiles_list):
    """
    Placeholder for extracting GNN-compatible graph features (atoms, bonds, adjacency).
    """
    raise NotImplementedError(
        "Graph feature extraction relies on PyTorch Geometric. "
        "Please ensure PyG is installed and graph extraction logic is plugged in."
    )

