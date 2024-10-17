import numpy as np

class ElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            intercept_old = self.intercept_

            # Update intercept
            self.intercept_ = np.mean(y - np.dot(X, self.coef_))

            # Update coefficients
            for j in range(n_features):
                residual = y - (self.intercept_ + np.dot(X, self.coef_))
                rho = np.dot(X[:, j], residual + self.coef_[j] * X[:, j])

                if rho < -self.alpha * self.l1_ratio:
                    self.coef_[j] = (rho + self.alpha * self.l1_ratio) / (np.dot(X[:, j], X[:, j]) + self.alpha * (1 - self.l1_ratio))
                elif rho > self.alpha * self.l1_ratio:
                    self.coef_[j] = (rho - self.alpha * self.l1_ratio) / (np.dot(X[:, j], X[:, j]) + self.alpha * (1 - self.l1_ratio))
                else:
                    self.coef_[j] = 0.0

            # Check for convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol and np.abs(self.intercept_ - intercept_old) < self.tol:
                break

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v    