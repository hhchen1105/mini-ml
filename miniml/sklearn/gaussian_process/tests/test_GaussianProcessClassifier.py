import pytest
import numpy as np
from miniml.sklearn.gaussian_process import GaussianProcessClassifier


def test_gpc_fit_and_basic_predict_proba():
    #binary classification dataset
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])

    gpc = GaussianProcessClassifier()
    gpc.fit(X, y)

    #   test sample
    X_test = np.array([[1.5]])
    proba = gpc.predict_proba(X_test)

    #output should have shape (n_test_samples, n_classes)
    assert proba.shape == (1, 2)

    #  values should be finite (no NaN or inf)
    assert np.isfinite(proba).all()

    # Probabilities for each sample should sum to 1
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(1))


def test_gpc_predict_returns_valid_labels():
    # Same small binary dataset
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])

    gpc = GaussianProcessClassifier()
    gpc.fit(X, y)

    #  test samples
    X_test = np.array([[0.1], [2.9]])
    y_pred = gpc.predict(X_test)

    # Number of predicted labels should match number of test samples
    assert y_pred.shape == (2,)

    # Every predicted label should be one of the known classes
    for label in y_pred:
        assert label in gpc.classes_


def test_gpc_multiclass_support():
    # 3-class dataset
    X = np.array([[0.0], [1.0], [2.0],
                  [3.0], [4.0], [5.0]])
    y = np.array([0, 1, 2, 0, 1, 2])

    gpc = GaussianProcessClassifier()
    gpc.fit(X, y)

    X_test = np.array([[0.5], [2.5], [4.5]])
    proba = gpc.predict_proba(X_test)

    assert proba.shape == (3, 3)

    # Probabilities must lie between 0 and 1
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)

    # each row of probabilities should sum to 1
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(3))


def test_gpc_handles_1d_input():
    # Passing X as a 1D array should still work
    X = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0, 1, 0, 1])

    gpc = GaussianProcessClassifier()
    gpc.fit(X, y)

    X_test = np.array([0.5, 2.5])
    proba = gpc.predict_proba(X_test)
    y_pred = gpc.predict(X_test)

    # Check that shapes are consistent
    assert proba.shape == (2, 2)
    assert y_pred.shape == (2,)


def test_gpc_raises_if_not_fitted():
 

    gpc = GaussianProcessClassifier()
    X_test = np.array([[1.0]])

    with pytest.raises(RuntimeError):
        gpc.predict(X_test)

 
    with pytest.raises(RuntimeError):
        gpc.predict_proba(X_test)


if __name__ == "__main__":
    pytest.main()
