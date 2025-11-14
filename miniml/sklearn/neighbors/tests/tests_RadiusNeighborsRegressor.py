import numpy as np
from miniml.sklearn.neighbors import RadiusNeighborsRegressor


def test_fit_returns_self():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])

    reg = RadiusNeighborsRegressor(radius=1.0)
    returned = reg.fit(X, y)

    assert returned is reg
    assert reg.X_.shape == X.shape
    assert reg.y_.shape == y.shape


def test_predict_with_neighbors_uses_mean():
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 2.0, 4.0])

    reg = RadiusNeighborsRegressor(radius=1.1)
    reg.fit(X_train, y_train)

    X_test = np.array([[0.9]])
    y_pred = reg.predict(X_test)

    assert y_pred.shape == (1,)
    assert np.isclose(y_pred[0], 1.0)


def test_predict_without_neighbors_uses_nearest():
    X_train = np.array([[0.0], [10.0]])
    y_train = np.array([1.0, 5.0])

    reg = RadiusNeighborsRegressor(radius=0.1)
    reg.fit(X_train, y_train)

    X_test = np.array([[9.9]])
    y_pred = reg.predict(X_test)

    assert np.isclose(y_pred[0], 5.0)


def test_predict_multiple_samples():
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 2.0])

    reg = RadiusNeighborsRegressor(radius=1.0)
    reg.fit(X_train, y_train)

    X_test = np.array([[0.0], [1.5], [3.0]])
    y_pred = reg.predict(X_test)

    assert y_pred.shape == (3,)
