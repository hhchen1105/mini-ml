import numpy as np


class Ridge():
    def __init__(self, alpha=1.0, *, fit_intercept=True, max_iter=None, tol=0.0001, solver='auto', random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state
        self.coef_, self.intercept_ = None, None

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        n, p = X.shape
        if self.solver == 'auto':
            self.solver = 'svd' if n > p else 'cholesky' if n > 2000 else 'sag'
        if self.solver == 'svd':
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            idx = s > 1e-10
            s_nnz = s[idx][:, np.newaxis]
            UTy = np.dot(U.T, y)
            self.coef_ = np.dot(Vt.T, UTy / s_nnz)
        elif self.solver == 'cholesky':
            L = np.linalg.cholesky(np.dot(X.T, X) + self.alpha * np.eye(p))
            self.coef_ = np.linalg.solve(L, np.dot(X.T, y))
        elif self.solver == 'sag':
            if self.random_state is not None:
                np.random.seed(self.random_state)
            alpha = self.alpha
            beta = 1.0 / (n * alpha)
            w = np.zeros(p)
            G = np.zeros(p)
            if self.fit_intercept:
                w[0] = np.mean(y)
                y = y - w[0]
            for _ in range(self.max_iter):
                i = np.random.randint(n)
                xi = X[i]
                yi = y[i]
                g = -2 * xi * (yi - np.dot(xi, w))
                G += g
                w -= beta * G
                w *= 1.0 / (1.0 + beta)
            self.coef_ = w
        self.coef_ = self.coef_
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        return self
    
    def predict(self, X):
        if self.fit_intercept:
            return np.dot(X, self.coef_) + self.intercept_
        else:
            return np.dot(X, self.coef_)
        
    def score(self, X, y):
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v