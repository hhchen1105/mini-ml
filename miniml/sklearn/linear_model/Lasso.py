import numpy as np
from miniml.sklearn.base import BaseEstimator

class Lasso(BaseEstimator):
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            for j in range(n_features):
                residual = y - (X @ self.coef_ + self.intercept_)
                rho = X[:, j] @ residual + self.coef_[j] * np.sum(X[:, j] ** 2)
                if rho < -self.alpha / 2:
                    self.coef_[j] = (rho + self.alpha / 2) / np.sum(X[:, j] ** 2)
                elif rho > self.alpha / 2:
                    self.coef_[j] = (rho - self.alpha / 2) / np.sum(X[:, j] ** 2)
                else:
                    self.coef_[j] = 0.0

            self.intercept_ = np.mean(y - X @ self.coef_)

            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v