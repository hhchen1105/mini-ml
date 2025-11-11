import numpy as np

class KNeighborsRegressor:
    def __init__(self, n_neighbors=5, weights='uniform'):
        if weights not in ('uniform', 'distance'):
            raise ValueError("weights must be 'uniform' or 'distance'")
        
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
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