import numpy as np

class SGDClassifier:
    def __init__(self, learning_rate=0.01, n_iter=1000, alpha=0.0001):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # 初始化權重
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.n_iter):
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                linear_output = np.dot(xi, self.coef_) + self.intercept_
                y_pred = self._sigmoid(linear_output)

                error = y_pred - yi

                # 更新權重 (加上 L2 正則化)
                self.coef_ -= self.learning_rate * (error * xi + self.alpha * self.coef_)
                self.intercept_ -= self.learning_rate * error

    def decision_function(self, X):
        X = np.array(X)
        return np.dot(X, self.coef_) + self.intercept_

    def predict_proba(self, X):
        scores = self.decision_function(X)
        probs = self._sigmoid(scores)
        return np.vstack([1 - probs, probs]).T  # [[p0, p1], ...]

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
