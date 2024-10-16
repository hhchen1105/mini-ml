import numpy as np


class Ridge:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = X.copy()
        y = y.copy()

        if self.fit_intercept:
            self.intercept_ = np.mean(y)
            y = y - self.intercept_
        else:
            self.intercept_ = 0.0

        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        d = s / (s**2 + self.alpha)
        self.coef_ = np.dot(Vt.T, d * np.dot(U.T, y))

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v
