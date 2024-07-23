import numpy as np


class LinearRegression():
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_, self.intercept_, self.rank_, self.singular_ = None, None, None, None

    def fit(self, X, y):
        self.coef_, self.intercept_, self.rank_, self.singular_ = np.linalg.lstsq(X, y, rcond=None)
        self.coef_ = self.coef_.T