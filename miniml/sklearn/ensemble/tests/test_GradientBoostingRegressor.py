import pytest
import numpy as np
from miniml.sklearn.ensemble.GradientBoostingRegressor import GradientBoostingRegressor

def test_gradient_boosting_regressor_fit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    assert len(model.trees) == 10

def test_gradient_boosting_regressor_predict():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape

def test_gradient_boosting_regressor_overfit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-1)

def test_gradient_boosting_regressor_underfit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = GradientBoostingRegressor(n_estimators=1, learning_rate=0.01, max_depth=1)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.allclose(predictions, y, atol=1e-1)

if __name__ == "__main__":
    pytest.main()