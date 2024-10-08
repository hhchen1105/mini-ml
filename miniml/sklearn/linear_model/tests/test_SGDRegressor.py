import pytest
import numpy as np
from miniml.sklearn.linear_model import SGDRegressor

def test_fit():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = SGDRegressor(learning_rate=0.01, n_iter=1000, alpha=0.0001)
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None

def test_predict():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = SGDRegressor(learning_rate=0.01, n_iter=1000, alpha=0.0001)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions is not None
    assert len(predictions) == len(y)

def test_score():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = SGDRegressor(learning_rate=0.01, n_iter=1000, alpha=0.0001)
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.9  # Assuming the model should fit well

def test_overfitting():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = SGDRegressor(learning_rate=0.01, n_iter=10000, alpha=0.0001)
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.99  # With more iterations, the model should fit even better

def test_underfitting():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = SGDRegressor(learning_rate=0.01, n_iter=5, alpha=0.0001)
    model.fit(X, y)
    score = model.score(X, y)
    assert score < 0.5  # With fewer iterations, the model might underfit

if __name__ == "__main__":
    pytest.main()
