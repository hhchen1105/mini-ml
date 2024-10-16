import pytest
import numpy as np
from miniml.sklearn.linear_model.Ridge import Ridge


def test_ridge_fit():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    assert model.coef_.shape == (2,)
    assert isinstance(model.intercept_, float)


def test_ridge_predict():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (3,)
    assert np.allclose(predictions, [1.90, 2.24, 2.59], atol=1e-1)


def test_ridge_score():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = Ridge(alpha=0.01)
    model.fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0.5 <= score <= 1


def test_ridge_fit_intercept():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(X, y)
    assert model.intercept_ == 0.0


if __name__ == "__main__":
    pytest.main()
