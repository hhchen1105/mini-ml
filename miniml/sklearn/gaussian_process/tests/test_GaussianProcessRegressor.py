import pytest
import numpy as np
from miniml.sklearn.gaussian_process import GaussianProcessRegressor


def test_gpr_fit_predict_mean():
    # Simple 1D dataset: y = sin(x)
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.sin(X).ravel()

    gpr = GaussianProcessRegressor()
    gpr.fit(X, y)

    X_test = np.array([[1.5]])
    y_pred = gpr.predict(X_test)

    # Prediction should be finite and shaped correctly
    assert np.isfinite(y_pred).all()
    assert y_pred.shape == (1,)


def test_gpr_return_std():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.sin(X).ravel()

    gpr = GaussianProcessRegressor()
    gpr.fit(X, y)

    X_test = np.array([[1.5]])
    y_mean, y_std = gpr.predict(X_test, return_std=True)

    # std must be positive
    assert y_std.shape == (1,)
    assert y_std[0] >= 0


def test_gpr_return_cov():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.sin(X).ravel()

    gpr = GaussianProcessRegressor()
    gpr.fit(X, y)

    X_test = np.array([[1.5], [2.0]])
    y_mean, y_cov = gpr.predict(X_test, return_cov=True)

    # Covariance matrix shape should be (n_test, n_test)
    assert y_cov.shape == (2, 2)

    # Covariance matrix should be symmetric
    assert np.allclose(y_cov, y_cov.T)

    # Diagonal (variance) should be >= 0
    assert np.all(np.diag(y_cov) >= 0)


def test_gpr_shape_handling():
    # Test X, y shapes like sklearn
    X = np.linspace(0, 5, 10).reshape(-1, 1)
    y = np.sin(X).ravel()  # shape (10,)

    gpr = GaussianProcessRegressor()
    gpr.fit(X, y)

    preds = gpr.predict(X)
    assert preds.shape == (10,)


def test_gpr_normalize_y_true():
    X = np.linspace(0, 3, 5).reshape(-1, 1)
    y = 3.0 + 2.0 * np.sin(X).ravel()   

    gpr = GaussianProcessRegressor(normalize_y=True)
    gpr.fit(X, y)

    X_test = np.array([[1.2], [2.5]])
    y_mean, y_std = gpr.predict(X_test, return_std=True)

    # y_std must be positive
    assert np.all(y_std >= 0)

    # y_mean should be finite and correct shape
    assert y_mean.shape == (2,)
    assert y_std.shape == (2,)


def test_gpr_covariance_matches_variance():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.sin(X).ravel()

    gpr = GaussianProcessRegressor()
    gpr.fit(X, y)

    X_test = np.array([[0.5], [1.5], [2.5]])

    # get covariance and std
    y_mean_cov, y_cov = gpr.predict(X_test, return_cov=True)
    y_mean_std, y_std = gpr.predict(X_test, return_std=True)

    # Extract variance from covariance diagonal
    variance_from_cov = np.diag(y_cov)

    # Compare std^2 with cov diagonal
    assert np.allclose(y_std ** 2, variance_from_cov, atol=1e-6)


if __name__ == "__main__":
    pytest.main()
