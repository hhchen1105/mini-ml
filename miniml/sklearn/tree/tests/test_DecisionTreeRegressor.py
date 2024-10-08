import pytest
import numpy as np
from miniml.sklearn.tree import DecisionTreeRegressor

def test_fit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)
    assert model.tree_ is not None

def test_predict():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 2, 3, 4, 4])
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, [2, 2, 3, 4, 4])

def test_predict_single_sample():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)
    prediction = model.predict(np.array([[3]]))
    assert np.allclose(prediction, [3])

def test_min_samples_split():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = DecisionTreeRegressor(min_samples_split=3)
    model.fit(X, y)
    assert model.tree_ is not None

def test_max_depth():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 2, 3, 4, 4])
    model = DecisionTreeRegressor(max_depth=1)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, [2, 2, 11/3, 11/3, 11/3])

def test_best_split():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 2, 3, 4, 4])
    model = DecisionTreeRegressor()
    feature, threshold, mse = model.best_split(X, y)
    assert feature == 0
    assert threshold == 2.
    assert mse < float('inf')

def test_split():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = DecisionTreeRegressor()
    mse = model.split(X, y, 0, 3)
    assert mse < float('inf')

def test_mse():
    y = np.array([1, 2, 3, 4, 5])
    model = DecisionTreeRegressor()
    mse = model.mse(y)
    assert mse == 2.0