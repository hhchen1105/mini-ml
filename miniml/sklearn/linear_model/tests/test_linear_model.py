from miniml.sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import Ridge as SKRidge
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


@pytest.mark.parametrize("alpha, fit_intercept, solver, random_state, intercept_, coef_, score", [
    (0, True, 'svd', 0, 3, np.array([1, 2]), 1),
    (0, False, 'svd', 0, 0., np.array([2.09090909, 2.54545454]), 0.748252),
    (1, True, 'svd', 0, 3, np.array([1, 2]), 1),
    (1, False, 'svd', 0, 0., np.array([2.09090909, 2.54545454]), 0.748252),
])
def test_Ridge(alpha, fit_intercept, solver, random_state, intercept_, coef_, score):
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver='svd')
    reg.fit(X, y)

    assert reg.intercept_ == pytest.approx(intercept_)
    assert_allclose(reg.coef_, coef_)
    assert reg.score(X, y) == pytest.approx(score)

    reg = Ridge(alpha=1., fit_intercept=fit_intercept, solver='svd')
    reg.fit(X, y)
    ridge = SKRidge(alpha=1., fit_intercept=fit_intercept, solver='svd').fit(X, y)
    assert reg.intercept_ == pytest.approx(ridge.intercept_)

    #assert 0 == 1
    # TODO: test when alpha=0 with diff solvers and diff fit_intercept
    # TODO: test when alpha=1 with diff solvers and diff fit_intercept