import numpy as np


class GaussianProcessClassifier:

    def __init__(self, kernel=None, alpha=1e-6, copy_X_train=True, random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.copy_X_train = copy_X_train
        self.random_state = random_state

        # Attributes set during fit
        self.classes_ = None
        self.n_classes_ = None
        self.X_train_ = None
        self.n_features_in_ = None
        self.kernel_ = None
        self._dual_coef = None  # shape (n_samples, n_classes)

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Inconsistent shapes: X has {X.shape[0]} samples "
                f"but y has {y.shape[0]}"
            )

        # Encode labels as indices 0..n_classes-1
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_in_ = X.shape[1]

        # Store training data
        self.X_train_ = X.copy() if self.copy_X_train else X

        # Choose kernel
        if self.kernel is None:
            self.kernel_ = self._rbf_kernel
        else:
            self.kernel_ = self.kernel

        # Kernel matrix on training data
        K = self.kernel_(self.X_train_, self.X_train_)

        # Regularization: K + alpha * I
        n_samples = K.shape[0]
        K_reg = K + self.alpha * np.eye(n_samples)

        # One-hot encode labels: shape (n_samples, n_classes)
        Y_onehot = self._one_hot(y_indices, self.n_classes_)

        # Solve K_reg @ dual_coef = Y_onehot
        self._dual_coef = np.linalg.solve(K_reg, Y_onehot)

        return self