import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class LogisticRegression:
    def __init__(self, penalty='l2', C=1.0, tol=1e-4, max_iter=100, fit_intercept=True):
        self.penalty = penalty
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, w, X, y):
        z = np.dot(X, w)
        h = self._sigmoid(z)
        loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        if self.penalty == 'l2':
            loss += 0.5 * self.C * np.sum(w ** 2)
        return loss

    def _gradient(self, w, X, y):
        z = np.dot(X, w)
        h = self._sigmoid(z)
        grad = np.dot(X.T, (h - y)) / y.size
        if self.penalty == 'l2':
            grad += self.C * w
        return grad

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        initial_w = np.zeros(X.shape[1])
        opt_w, _, _ = fmin_l_bfgs_b(self._loss, initial_w, fprime=self._gradient, args=(X, y), pgtol=self.tol, maxiter=self.max_iter)
        
        if self.fit_intercept:
            self.intercept_ = opt_w[0]
            self.coef_ = opt_w[1:]
        else:
            self.intercept_ = 0
            self.coef_ = opt_w

    def predict_proba(self, X):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        z = np.dot(X, np.hstack([self.intercept_, self.coef_]))
        return self._sigmoid(z)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    