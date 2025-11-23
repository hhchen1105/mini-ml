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