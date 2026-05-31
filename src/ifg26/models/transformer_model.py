"""
transformer_model.py

Transformer baseline model (e.g. ChemBERTa or similar SMILES transformer).
"""
from ifg26.models.base_model import BaseModel
import numpy as np

class TransformerModel(BaseModel):
    """
    A text-based Transformer for SMILES embeddings.
    Fails gracefully if HuggingFace transformers library is unavailable.
    """
    
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MTR", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        
        try:
            import transformers
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            raise ImportError(
                "transformers or torch are missing. Please install the Hugging Face "
                "transformers library to use the Transformer baseline."
            )
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        print(f"STUB: Fine-tuning TransformerModel ({self.model_name}).")
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lg_msg = (
            f"TransformerModel ({self.model_name}) prediction failed: The SMILES Transformer baseline is currently a non-functional placeholder stub. "
            "SILENT RANDOM PREDICTIONS ARE BLOCKED to prevent scientific metric contamination. "
            "Please install HuggingFace transformers and download model weights to use this baseline."
        )
        print(f"ERROR: {lg_msg}")
        raise NotImplementedError(lg_msg)

if __name__ == "__main__":
    print("Running Smoke Test for transformer_model.py...")
    try:
        model = TransformerModel()
        model.fit(np.array(["C"]), np.array([0]))
        preds = model.predict_proba(np.array(["C"]))
        assert len(preds) == 1
        print("Smoke Test Passed!")
    except ImportError as e:
        print(f"Smoke Test Skipped (Expected if transformers is missing): {e}")
