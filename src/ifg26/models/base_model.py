"""
base_model.py

Abstract base class for all benchmark models.
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Standard interface for all IFG-26 Benchmark models.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the training data.
        
        Args:
            X: Training features (N_samples, N_features)
            y: Training labels (N_samples,)
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of glue activity.
        
        Args:
            X: Test features (N_samples, N_features)
            
        Returns:
            np.ndarray: Predicted probabilities (N_samples,) showing probability of positive class.
        """
        pass
