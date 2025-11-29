import numpy as np
from ..tree.DecisionTreeRegressor import DecisionTreeRegressor
from miniml.sklearn.base import BaseEstimator


class RandomForestRegressor(BaseEstimator):
    def __init__(
        self, n_estimators=100, max_features="sqrt", max_depth=None, random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _get_max_features(self, n_features):
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            self.feature_indices.append(feature_indices)
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array(
            [
                tree.predict(X[:, feature_indices])
                for tree, feature_indices in zip(self.trees, self.feature_indices)
            ]
        )
        return np.mean(tree_predictions, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)