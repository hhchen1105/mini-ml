import numpy as np
import pytest
from miniml.sklearn.neighbors import RadiusNeighborsRegressor


def test_fit_returns_self():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])

    reg = RadiusNeighborsRegressor(radius=1.0)
    returned = reg.fit(X, y)

    assert returned is reg
    assert reg.X_.shape == X.shape
    assert reg.y_.shape == y.shape


def test_predict_with_neighbors_uses_mean_uniform():
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 2.0, 4.0])

    reg = RadiusNeighborsRegressor(radius=1.1, weights="uniform")
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


def test_predict_multiple_samples_shape():
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 2.0])

    reg = RadiusNeighborsRegressor(radius=1.0)
    reg.fit(X_train, y_train)

    X_test = np.array([[0.0], [1.5], [3.0]])
    y_pred = reg.predict(X_test)

    assert y_pred.shape == (3,)


def test_invalid_radius_raises_value_error():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        RadiusNeighborsRegressor(radius=0.0)

    with pytest.raises(ValueError):
        RadiusNeighborsRegressor(radius=-1.0)


def test_zero_distance_returns_exact_target_with_distance_weights():
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([1.0, 3.0])

    reg = RadiusNeighborsRegressor(radius=1.0, weights="distance")
    reg.fit(X_train, y_train)

    X_test = np.array([[1.0]])
    y_pred = reg.predict(X_test)

    assert np.isclose(y_pred[0], 3.0)


def test_distance_weighting_gives_closer_points_more_influence():
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0.0, 2.0])

    reg_uniform = RadiusNeighborsRegressor(radius=2.0, weights="uniform")
    reg_uniform.fit(X_train, y_train)

    reg_distance = RadiusNeighborsRegressor(radius=2.0, weights="distance")
    reg_distance.fit(X_train, y_train)

    X_test = np.array([[0.1]])
    y_pred_uniform = reg_uniform.predict(X_test)[0]
    y_pred_distance = reg_distance.predict(X_test)[0]

    assert np.isclose(y_pred_uniform, 1.0)
    assert y_pred_distance < y_pred_uniform


def test_multi_output_regression_supported():
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    reg = RadiusNeighborsRegressor(radius=1.5, weights="uniform")
    reg.fit(X_train, y_train)

    X_test = np.array([[1.0]])
    y_pred = reg.predict(X_test)

    assert y_pred.shape == (1, 2)
    assert np.allclose(y_pred[0], np.array([1.0, 2.0]))


def test_predict_feature_dimension_mismatch_raises():
    X_train = np.array([[0.0, 1.0], [1.0, 2.0]])
    y_train = np.array([0.0, 1.0])

    reg = RadiusNeighborsRegressor(radius=1.0)
    reg.fit(X_train, y_train)

    X_test = np.array([[0.0]])
    with pytest.raises(ValueError):
        reg.predict(X_test)
