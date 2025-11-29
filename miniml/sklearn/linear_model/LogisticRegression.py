import numpy as np
from scipy.optimize import minimize
from miniml.sklearn.base import BaseEstimator

class LogisticRegression(BaseEstimator):
    def __init__(self, C=1.0, max_iter=100, tol=1e-4):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        if self.C <= 0:
            raise ValueError("Regularization parameter C is the inverse of the regularization strength; must be positive.")

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, w, X, y):
        m = X.shape[0]
        h = self._sigmoid(X @ w)
        reg = (1 / (2 * self.C)) * np.sum(w[1:] ** 2)
        return (-1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h)) + reg

    def _gradient(self, w, X, y):
        m = X.shape[0]
        h = self._sigmoid(X @ w)
        grad = (1 / m) * (X.T @ (h - y))
        grad[1:] += (1 / self.C) * w[1:]
        return grad

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add intercept term
        initial_w = np.zeros(X.shape[1])
        options = {'maxiter': self.max_iter, 'disp': False}

        res = minimize(fun=self._loss, x0=initial_w, args=(X, y), method='L-BFGS-B', jac=self._gradient, options=options, tol=self.tol)

        self.coef_ = res.x[1:]
        self.intercept_ = res.x[0]

    def predict_proba(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add intercept term
        return self._sigmoid(X @ np.insert(self.coef_, 0, self.intercept_))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)