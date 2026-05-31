"""
random_forest.py

Random Forest baseline model. 
"""
from ifg26.models.base_model import BaseModel
import numpy as np

class RandomForestModel(BaseModel):
    
    def __init__(self, n_estimators: int = 500, max_depth: int = None, random_state: int = 42):
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        except Exception:
            self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise NotImplementedError("RandomForest requires sklearn, which is broken on this host.")
        self.model.fit(X, y)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise NotImplementedError("RandomForest requires sklearn, which is broken on this host.")
        return self.model.predict_proba(X)[:, 1]

if __name__ == "__main__":
    print("Smoke Test for random_forest.py")
    m = RandomForestModel(n_estimators=10)
    m.fit(np.array([[0], [1]]), np.array([0, 1]))
    p = m.predict_proba(np.array([[1]]))
    assert len(p) == 1
    print("Passed!")
