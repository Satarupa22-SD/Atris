from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import numpy as np

class KNNOutlierAnomaly(BaseEstimator):
    def __init__(self, n_neighbors=5, threshold=3.0):
        self.n_neighbors = n_neighbors
        self.threshold = threshold
    def fit(self, X, y=None):
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nbrs_.fit(X)
        distances, _ = self.nbrs_.kneighbors(X)
        self.mean_dist_ = np.mean(distances[:, 1:])  # skip self-distance
        self.std_dist_ = np.std(distances[:, 1:])
        return self
    def predict(self, X):
        distances, _ = self.nbrs_.kneighbors(X)
        mean_dist = np.mean(distances[:, 1:], axis=1)
        return (mean_dist > self.mean_dist_ + self.threshold * self.std_dist_).astype(int) 