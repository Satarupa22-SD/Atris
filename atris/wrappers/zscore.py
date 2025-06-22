from sklearn.base import BaseEstimator
import numpy as np

class ZScoreAnomaly(BaseEstimator):
    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
    def predict(self, X):
        z = (X - self.mean_) / (self.std_ + 1e-8)
        # 1 = anomaly, 0 = normal
        return (np.abs(z) > 3).any(axis=1).astype(int) 