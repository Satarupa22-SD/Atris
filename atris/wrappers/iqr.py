from sklearn.base import BaseEstimator
import numpy as np

class IQRAnomaly(BaseEstimator):
    def fit(self, X, y=None):
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        return self
    def predict(self, X):
        lower = self.q1_ - 1.5 * self.iqr_
        upper = self.q3_ + 1.5 * self.iqr_
        outlier = ((X < lower) | (X > upper)).any(axis=1)
        return outlier.astype(int) 