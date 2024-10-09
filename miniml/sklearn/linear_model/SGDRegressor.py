import numpy as np


class SGDRegressor:
    def __init__(self, learning_rate=0.01, n_iter=1000, alpha=0.0001):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.n_iter):
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                prediction = np.dot(xi, self.coef_) + self.intercept_
                error = prediction - yi

                self.coef_ -= self.learning_rate * (
                    error * xi + self.alpha * self.coef_
                )
                self.intercept_ -= self.learning_rate * error

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v
