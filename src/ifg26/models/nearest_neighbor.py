"""
nearest_neighbor.py

Nearest neighbor baseline using KNeighborsClassifier to test similarity search bias.
"""
from ifg26.models.base_model import BaseModel
import numpy as np

class NearestNeighborModel(BaseModel):
    
    def __init__(self, n_neighbors: int = 5, metric: str = 'jaccard'):
        try:
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric=metric,
                n_jobs=-1
            )
        except Exception:
            self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise NotImplementedError("NearestNeighbor requires sklearn, which is broken.")
        self.model.fit(X, y)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise NotImplementedError("NearestNeighbor requires sklearn, which is broken.")
        return self.model.predict_proba(X)[:, 1]

if __name__ == "__main__":
    print("Smoke Test for nearest_neighbor.py")
    m = NearestNeighborModel(n_neighbors=1)
    m.fit(np.array([[0], [1]]), np.array([0, 1]))
    p = m.predict_proba(np.array([[1]]))
    assert len(p) == 1
    print("Passed!")
