import numpy as np
from miniml.sklearn.base import BaseEstimator


class ExtraTreeRegressor(BaseEstimator):
    """
    A minimal implementation of ExtraTreeRegressor for mini-ml homework.
    This version only learns the mean target value and predicts the same
    value for all inputs.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state
        self._is_fitted = False
        self._y_mean = None

    def fit(self, X, y):
        """
        Fit the model by storing the mean of y.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (not actually used in this simple model).
        y : array-like, shape (n_samples,)
            Target values.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self._y_mean = float(np.mean(y))
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Predict by repeating the stored mean value for each sample in X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        X = np.asarray(X)
        n_samples = X.shape[0]
        return np.full(n_samples, self._y_mean, dtype=float)
