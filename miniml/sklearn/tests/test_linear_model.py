from miniml.sklearn.linear_model import LinearRegression
import numpy as np
import pytest
from numpy.testing import assert_allclose

def test_LinearRegression():
    # test LinearRegression with fit_intercept=True
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression()
    reg.fit(X, y)
    assert reg.intercept_ == pytest.approx(3)
    assert_allclose(reg.coef_, np.array([1, 2]))

    # TODO: test LinearRegression with fit_intercept=False
    #       it seems pytest supports parameterize function; see
    #       https://stackoverflow.com/questions/70619728/testing-the-same-function-with-different-parameters

    # TODO: Does it work if we use reg = LinearRegression.fit(X, y)?
