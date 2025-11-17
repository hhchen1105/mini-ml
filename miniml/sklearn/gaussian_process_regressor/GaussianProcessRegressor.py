import numpy as np

class GaussianProcessRegressor:
    def __init__ (self, kernel=None, *, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, 
                normalize_y=False, copy_X_train=True, n_targets=None, random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.n_targets = n_targets
        self.random_state = random_state
        self.kernel_ = None
        self.alpha_ = None
        self.X_train_ = self.y_train_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self.L_ = None

    def fit(self, X, y):
        # Ensure correct shapes
        X = self._ensure_2d(X)
        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)  # handle single target

        n_samples = X.shape[0]

        # Save training data
        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = y

        # Normalize y if needed
        if self.normalize_y:
            self.y_mean_ = y.mean(axis=0)
            self.y_std_  = y.std(axis=0)
            y = (y - self.y_mean_) / self.y_std_
        else:
            self.y_mean_ = 0
            self.y_std_ = 1

        # Select kernel
        if self.kernel is None:  # If kernel=None, use default ConstantKernel(1.0) * RBF(1.0)
            self.kernel_ = lambda a, b: self._rbf_kernel(a, b)
        else:
            self.kernel_ = self.kernel

        # Compute Gram matrix K
        K = self.kernel_(X, X)

        # Add noise term alpha * I
        K += self.alpha * np.eye(n_samples)

        # Cholesky decomposition
        self.L_ = np.linalg.cholesky(K)

        # Solve for alpha = (K + αI)^(-1) y
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, y))

        return self
    
    def predict(self, X, return_std=False, return_cov=False):
        # Safety check: std and cov cannot both be True
        if return_std and return_cov:
            raise ValueError("Cannot return both std and cov. Choose one.")

        # Format X
        X = self._ensure_2d(X)

        # Compute kernel between test and training points
        K_trans = self.kernel_(X, self.X_train_)   # shape: (n_test, n_train)

        # Predictive mean: K_* @ alpha
        y_mean = K_trans.dot(self.alpha_)

        # Undo normalization if needed
        if self.normalize_y:
            y_mean = y_mean * self.y_std_ + self.y_mean_

        # If only mean is needed, return here
        if not return_std and not return_cov:
            return y_mean.ravel()  # flatten like sklearn

        # To compute std or cov, compute v = solve(L, K_*^T)
        # Solve L v = K_*^T
        v = np.linalg.solve(self.L_, K_trans.T)

        # Return covariance
        if return_cov:
            # K** = kernel(X, X)
            K_self = self.kernel_(X, X)  # shape (n_test, n_test)
            y_cov = K_self - v.T.dot(v)
            return y_mean.ravel(), y_cov

        # Return standard deviation
        # diagonal of covariance = k(x*, x*) - sum(v**2)
        K_self_diag = np.diag(self.kernel_(X, X))
        y_var = K_self_diag - np.sum(v**2, axis=0)
        y_std = np.sqrt(np.maximum(y_var, 0))  # numerical stability

        return y_mean.ravel(), y_std


    def _rbf_kernel(self, X1, X2, length_scale=1.0, sigma_f=1.0):
        """Simplified RBF kernel: sigma_f^2 * exp(-||x-x'||^2/(2*l^2))"""
        sqdist = (
            np.sum(X1**2, axis=1).reshape(-1,1)
            + np.sum(X2**2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )
        return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X