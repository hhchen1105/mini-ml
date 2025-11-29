import numpy as np
from ..tree.DecisionTreeRegressor import DecisionTreeRegressor
from miniml.sklearn.base import BaseEstimator

class ExtraTreesRegressor(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt",
                 bootstrap=False, random_state=None):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []  
        self._rng = np.random.default_rng(random_state)

    def _resolve_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if self.max_features == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError("Unsupported string for max_features")
        if isinstance(self.max_features, (int, np.integer)):
            return min(int(self.max_features), n_features)
        if isinstance(self.max_features, (float, np.floating)):
            return max(1, int(np.ceil(self.max_features * n_features)))
        raise TypeError("Invalid type for max_features")

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        m = self._resolve_max_features(n_features)
        self.trees = []

        for _ in range(self.n_estimators):
            # Randomly select a subset of features
            feature_indices = self._rng.choice(n_features, size=m, replace=False)

            if self.bootstrap:
                sample_indices = self._rng.choice(n_samples, size=n_samples, replace=True)
                X_sub = X[sample_indices][:, feature_indices]
                y_sub = y[sample_indices]
            else:
                X_sub = X[:, feature_indices]
                y_sub = y

            # Train one decision tree on selected features and samples
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_sub, y_sub)

            # Store the trained tree and its feature subset
            self.trees.append((tree, feature_indices))

        return self

    def predict(self, X):
        X = np.asarray(X)
        if not self.trees:
            raise RuntimeError("Model is not fitted yet.")

        # Average predictions from all trees
        agg = np.zeros(X.shape[0], dtype=float)
        for tree, feature_indices in self.trees:
            agg += tree.predict(X[:, feature_indices])
        return agg / len(self.trees)
