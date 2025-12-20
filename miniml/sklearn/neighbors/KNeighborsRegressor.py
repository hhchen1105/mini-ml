import numpy as np
from miniml.sklearn.base import BaseEstimator

class KNeighborsRegressor(BaseEstimator):
    def __init__(self, n_neighbors=5, weights='uniform'):
        if weights not in ('uniform', 'distance'):
            raise ValueError("weights must be 'uniform' or 'distance'")
        
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        if self.n_neighbors > len(self.y_train):
            raise ValueError(f"n_neighbors ({self.n_neighbors}) is greater than number of training samples ({len(self.y_train)}).")

    def predict(self, X):
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(f"Number of features in X ({X.shape[1]}) does not match training data ({self.X_train.shape[1]}).")
        
        predictions = [self._predict_sample(x) for x in X]
        return np.array(predictions)

    def _predict_sample(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_values = self.y_train[k_indices]
        neighbor_dist = distances[k_indices]

        if self.weights == 'uniform':
            return np.mean(k_nearest_values)
        else:
            if np.any(neighbor_dist == 0):
                return k_nearest_values[neighbor_dist == 0][0]
            else:
                weights = 1 / neighbor_dist
                return np.sum(weights * k_nearest_values) / np.sum(weights)