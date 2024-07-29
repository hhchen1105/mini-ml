from miniml.sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression as SKLinearRegression
import numpy as np
import pytest
from numpy.testing import assert_allclose

@pytest.mark.parametrize("fit_intercept, intercept_, coef_, score", [
    (True, 3, np.array([1, 2]), 1.),
    (False, 0., np.array([2.09090909, 2.54545454]), 0.748252),
])
def test_LinearRegression(fit_intercept, intercept_, coef_, score):
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X, y)

    # test fitted intercept_ and coef_
    assert reg.intercept_ == pytest.approx(intercept_)
    assert_allclose(reg.coef_, coef_)

    # test LinearRegression.score(X, y)
    assert reg.score(X, y) == pytest.approx(score)

    # test X's shape after fitting
    assert X.shape == (4, 2)


def test_Ridge():
    assert 0 == 1
    # TODO: test when alpha=0 with diff solvers and diff fit_intercept
    # TODO: test when alpha=1 with diff solvers and diff fit_intercept