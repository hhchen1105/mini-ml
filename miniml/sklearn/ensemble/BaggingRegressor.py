import numpy as np
from ..tree.DecisionTreeRegressor import DecisionTreeRegressor

class BaggingRegressor:
    def __init__(self, n_estimators=10, max_samples=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.trees = [] # 命名為 self.trees，與範例檔案一致

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        # 根據 self.max_samples 決定抽樣數量
        n_bootstrap_samples = int(self.max_samples * n_samples)
        
        # np.random.choice 會幫我們處理隨機抽樣
        # replace=True 是 Bagging 的核心
        indices = np.random.choice(n_samples, n_bootstrap_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.trees = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor() 
            X_sample, y_sample = self._bootstrap_sample(X, y)
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