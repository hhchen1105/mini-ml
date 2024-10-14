import pytest
import numpy as np
from miniml.sklearn.ensemble.RandomForestRegressor import RandomForestRegressor

@pytest.fixture
def data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    return X, y

def test_random_forest_regressor_fit(data):
    X, y = data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    assert len(model.trees) == 10
    assert len(model.feature_indices) == 10

def test_random_forest_regressor_predict(data):
    X, y = data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape

def test_random_forest_regressor_score(data):
    X, y = data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    score = model.score(X, y)
    assert 0 <= score <= 1

def test_random_forest_regressor_different_max_features(data):
    X, y = data
    model = RandomForestRegressor(n_estimators=10, max_features=1, random_state=42)
    model.fit(X, y)
    assert all(len(indices) == 1 for indices in model.feature_indices)