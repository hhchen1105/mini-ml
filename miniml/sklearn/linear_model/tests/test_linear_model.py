from miniml.sklearn.linear_model import LinearRegression
import numpy as np
import pytest
from numpy.testing import assert_allclose

@pytest.mark.parametrize("fit_intercept, intercept, coef_", [
    (True, 3, np.array([1, 2])),
    (False, 0., np.array([2.09090909, 2.54545454])),
])
def test_LinearRegression(fit_intercept, intercept, coef_):
    # test fit_intercept == True or False
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X, y)
    assert reg.intercept_ == pytest.approx(intercept)
    assert_allclose(reg.coef_, coef_)

    # test the usage of reg = LinearRegression.fit(X, y)?
    reg = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    assert reg.intercept_ == pytest.approx(intercept)
    assert_allclose(reg.coef_, coef_)
