from miniml.sklearn.linear_model import Ridge
from sklearn.linear_model import Ridge as SKRidge
import numpy as np
import pytest
from numpy.testing import assert_allclose


#@pytest.mark.parametrize("alpha, fit_intercept, solver, random_state, intercept_, coef_, score", [
#    (0, True, 'svd', 0, 3, np.array([1, 2]), 1),
#    (0, False, 'svd', 0, 0., np.array([2.09090909, 2.54545454]), 0.748252),
#    (1, True, 'svd', 0, 3, np.array([1, 2]), 1),
#    (1, False, 'svd', 0, 0., np.array([2.09090909, 2.54545454]), 0.748252),
#])
#def test_Ridge(alpha, fit_intercept, solver, random_state, intercept_, coef_, score):
def test_Ridge():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    alpha, fit_intercept, solver = 0, True, 'svd'
    reg = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver)
    reg.fit(X, y)

    assert reg.intercept_ == pytest.approx(3)
    assert_allclose(reg.coef_, np.array([1, 2]))
    assert reg.score(X, y) == pytest.approx(1)

    #reg = Ridge(alpha=1., fit_intercept=fit_intercept, solver='svd')
    #reg.fit(X, y)
    #ridge = SKRidge(alpha=1., fit_intercept=fit_intercept, solver='svd').fit(X, y)
    #assert reg.intercept_ == pytest.approx(ridge.intercept_)

    #assert 0 == 1
    # TODO: test when alpha=0 with diff solvers and diff fit_intercept
    # TODO: test when alpha=1 with diff solvers and diff fit_intercept