"""
gnn_model.py

Graph Neural Network MVP baseline class.
"""
from ifg26.models.base_model import BaseModel
import numpy as np

class GNNModel(BaseModel):
    """
    A Graph Neural Network baseline, e.g. using PyTorch Geometric.
    Currently a stub that fails gracefully if PyTorch is not available.
    """
    
    def __init__(self, hidden_dim: int = 64, lr: float = 1e-3, epochs: int = 10):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        
        try:
            import torch
            import torch.nn as nn
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_initialized = True
        except ImportError:
            raise ImportError(
                "PyTorch is missing. Please install torch and torch-geometric "
                "to use the GNN baseline model."
            )
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        print(f"STUB: Training GNNModel for {self.epochs} epochs.")
        # Expect X to be a list of PyG Data objects or similar, handled by featurizer.
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lg_msg = (
            "GNNModel prediction failed: The GNN baseline is currently a non-functional placeholder stub. "
            "SILENT RANDOM PREDICTIONS ARE BLOCKED to prevent scientific metric contamination. "
            "Please install PyTorch Geometric and load genuine pre-trained GNN weights to evaluate this baseline."
        )
        print(f"ERROR: {lg_msg}")
        raise NotImplementedError(lg_msg)
        
if __name__ == "__main__":
    print("Running Smoke Test for gnn_model.py...")
    try:
        model = GNNModel()
        model.fit(np.array([[0]]), np.array([0]))
        preds = model.predict_proba(np.array([[0]]))
        assert len(preds) == 1
        print("Smoke Test Passed!")
    except ImportError as e:
        print(f"Smoke Test Skipped (Expected if torch is missing): {e}")
