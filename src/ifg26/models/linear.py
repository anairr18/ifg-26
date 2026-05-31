"""
linear.py

Linear baseline model using Logistic Regression.
"""
from ifg26.models.base_model import BaseModel
import numpy as np

class LinearModel(BaseModel):
    
    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        try:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=-1
            )
        except Exception:
            self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise NotImplementedError("LinearModel requires sklearn, which is broken.")
        self.model.fit(X, y)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise NotImplementedError("LinearModel requires sklearn, which is broken.")
        return self.model.predict_proba(X)[:, 1]

if __name__ == "__main__":
    print("Smoke Test for linear.py")
    m = LinearModel(max_iter=10)
    m.fit(np.array([[0], [1]]), np.array([0, 1]))
    p = m.predict_proba(np.array([[1]]))
    assert len(p) == 1
    print("Passed!")
