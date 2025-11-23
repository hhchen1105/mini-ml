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
        self._dual_coef = None 

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Inconsistent shapes: X has {X.shape[0]} samples "
                f"but y has {y.shape[0]}"
            )

        # Get the unique class labels and map them to integer indices
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_in_ = X.shape[1]

        # a copy of the training data
        self.X_train_ = X.copy() if self.copy_X_train else X

        # Decide which kernel function to use
        if self.kernel is None:
            self.kernel_ = self._rbf_kernel
        else:
            self.kernel_ = self.kernel

        # Compute the kernel matrix on the training set
        K = self.kernel_(self.X_train_, self.X_train_)

        # Add a small value on the diagonal to make the system more stable
        n_samples = K.shape[0]
        K_reg = K + self.alpha * np.eye(n_samples)

        # Turn class indices into a one-hot representation
        Y_onehot = self._one_hot(y_indices, self.n_classes_)

        # Find the coefficients that link training points to class scores
        self._dual_coef = np.linalg.solve(K_reg, Y_onehot)

        return self

    def predict_proba(self, X):
        self._check_is_fitted()

        X = self._prepare_X(X)

        #Measure how similar each test point is to the training points
        K_trans = self.kernel_(X, self.X_train_)  # (n_test, n_train)

        # Turn similarities into raw scores for each class
        scores = K_trans @ self._dual_coef

        #Convert scores into probabilities (softmax)
        scores = scores - scores.max(axis=1, keepdims=True)  # avoid overflow
        exp_scores = np.exp(scores)
        proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X):
        #Use the class with the highest predicted probability
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

    def _prepare_X(self, X):
        # Convert X to a 2D NumPy array.
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    @staticmethod
    def _one_hot(y, n_classes):
        # Create a one-hot encoding
        y = np.asarray(y, dtype=int)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of label indices.")
        n_samples = y.shape[0]
        Y = np.zeros((n_samples, n_classes), dtype=float)
        Y[np.arange(n_samples), y] = 1.0
        return Y

    @staticmethod
    def _rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
        # RBF kernel based on squared Euclidean distance.
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        sq_norms_1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sq_norms_2 = np.sum(X2 ** 2, axis=1)
        sqdist = sq_norms_1 + sq_norms_2 - 2 * X1 @ X2.T
        return (sigma_f ** 2) * np.exp(-0.5 * sqdist / (length_scale ** 2))

    def _check_is_fitted(self):
        if self._dual_coef is None or self.X_train_ is None or self.classes_ is None:
            raise RuntimeError(
                "GaussianProcessClassifier is not fitted yet. "
                "Call 'fit' before calling 'predict' or 'predict_proba'."
            )
