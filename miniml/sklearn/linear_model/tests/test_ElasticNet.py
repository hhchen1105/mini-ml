import pytest
import numpy as np
from miniml.sklearn.linear_model import ElasticNet

def test_elasticnet_fit():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4)
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None

def test_elasticnet_predict():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions is not None
    assert len(predictions) == len(y)

def test_elasticnet_score():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4)
    model.fit(X, y)
    score = model.score(X, y)
    assert 0.8 <= score <= 1

def test_elasticnet_convergence():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10, tol=1e-4)
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None

def test_elasticnet_zero_alpha():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])
    model = ElasticNet(alpha=0.0, l1_ratio=0.5, max_iter=1000, tol=1e-4)
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None

if __name__ == "__main__":
    pytest.main()