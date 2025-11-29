import numpy as np
import copy
from ..tree.DecisionTreeRegressor import DecisionTreeRegressor
from miniml.sklearn.base import BaseEstimator

class BaggingRegressor(BaseEstimator):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.trees = []
        self.base_estimator_ = None

    def _bootstrap_sample(self, X, y, rng):
        n_samples = X.shape[0]
        # 根據 self.max_samples 決定抽樣數量
        n_bootstrap_samples = int(self.max_samples * n_samples)
        
        # use rng.choice
        indices = rng.choice(n_samples, n_bootstrap_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        # Boundary testing
        if not (0.0 < self.max_samples <= 1.0):
            raise ValueError(f"max_samples must be in (0.0, 1.0], but got {self.max_samples}")
        # Create a local RandomState generator
        rng = np.random.RandomState(self.random_state)

        self.trees = []

        if self.base_estimator is None:
            self.base_estimator_ = DecisionTreeRegressor()
        else:
            self.base_estimator_ = self.base_estimator
        
        self.trees = []

        for _ in range(self.n_estimators):
            tree = copy.deepcopy(self.base_estimator_)
            X_sample, y_sample = self._bootstrap_sample(X, y, rng)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # 收集所有樹的預測值
        tree_predictions = np.array(
            [tree.predict(X) for tree in self.trees]
        )
        
        # 沿著 axis=0 (即「模型」的維度) 取平均
        return np.mean(tree_predictions, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)