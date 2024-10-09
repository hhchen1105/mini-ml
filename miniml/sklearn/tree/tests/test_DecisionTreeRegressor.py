import numpy as np
from miniml.sklearn.tree import DecisionTreeRegressor


def test_fit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)
    assert model.tree is not None


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


def test_max_depth():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 2, 3, 4, 4])
    model = DecisionTreeRegressor(max_depth=1)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, [2, 2, 11 / 3, 11 / 3, 11 / 3])


def test_best_split():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 2, 3, 4, 4])
    model = DecisionTreeRegressor()
    feature, threshold, (X_left, X_right, y_left, y_right) = model._best_split(X, y)
    assert feature == 0
    assert threshold == 2.0
    assert np.allclose(X_left, [[1], [2]])
    assert np.allclose(X_right, [[3], [4], [5]])
    assert np.allclose(y_left, [2, 2])
    assert np.allclose(y_right, [3, 4, 4])


def test_split():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = DecisionTreeRegressor()
    X_left, X_right, y_left, y_right = model._split(X, y, 0, 3)
    np.allclose(X_left, [[1], [2], [3]])
    np.allclose(X_right, [[4], [5]])
    np.allclose(y_left, [1, 2, 3])
    np.allclose(y_right, [4, 5])


def test_mse():
    y = np.array([1, 2, 3, 4, 5])
    model = DecisionTreeRegressor()
    mse = model._mse(y)
    assert mse == 2.0
