import pytest
import numpy as np
from miniml.sklearn.ensemble.BaggingRegressor import BaggingRegressor

@pytest.fixture
def data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    return X, y

def test_bagging_regressor_fit(data):
    X, y = data
    n_estimators = 10 
    model = BaggingRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    
    assert len(model.trees) == n_estimators

def test_bagging_regressor_predict(data):
    X, y = data
    model = BaggingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape

def test_bagging_regressor_score(data):
    X, y = data
    model = BaggingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    score = model.score(X, y)

    assert score <= 1.0 
    assert score > -np.inf 


def test_bagging_regressor_max_samples(data):
    X, y = data
    model = BaggingRegressor(n_estimators=10, max_samples=0.5, random_state=42)   

    X_sample, y_sample = model._bootstrap_sample(X, y)
    
    expected_samples = int(0.5 * X.shape[0])
    
    assert X_sample.shape[0] == expected_samples
    assert y_sample.shape[0] == expected_samples