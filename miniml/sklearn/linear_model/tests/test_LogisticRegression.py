import pytest
import numpy as np
from miniml.sklearn.linear_model import LogisticRegression

def test_sigmoid():
    model = LogisticRegression()
    z = np.array([0, 2, -2])
    expected = np.array([0.5, 0.88079708, 0.11920292])
    np.testing.assert_almost_equal(model._sigmoid(z), expected, decimal=6)

def test_loss():
    model = LogisticRegression(C=1.0)
    X = np.array([[1, 2], [1, -1], [1, 1]])
    y = np.array([1, 0, 1])
    w = np.array([0.5, -0.5])
    h = 1./(1+np.exp(-np.dot(X, w)))
    expected_loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)) + 1 / (2 * 1.0) * ((-0.5)**2)

    np.testing.assert_almost_equal(model._loss(w, X, y), expected_loss, decimal=6)

def test_gradient():
    model = LogisticRegression(C=1.0)
    X = np.array([[1, 2], [1, -1], [1, 1]])
    y = np.array([1, 0, 1])
    w = np.array([0.5, -0.5])
    expected_grad = np.dot(X.T, (1./(1+np.exp(-np.dot(X, w))) - y)) / y.size + 1.0 * np.array([0, -0.5])
    np.testing.assert_almost_equal(model._gradient(w, X, y), expected_grad, decimal=6)

def test_fit():
    model = LogisticRegression(max_iter=200, C=1.)
    X = np.array([[1, 2], [1, -1], [2, 1], [2, 2], [3, -1]])
    y = np.array([1, 0, 1, 1, 0])
    model.fit(X, y)
    assert model.coef_.shape == (2,)
    assert isinstance(model.intercept_, float)

def test_predict_proba():
    model = LogisticRegression(max_iter=200)
    X = np.array([[1, 2], [1, -1], [2, 1], [2, 2], [3, -1]])
    y = np.array([1, 0, 1, 1, 0])
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (5,)
    assert np.all(proba >= 0) and np.all(proba <= 1)

def test_predict_log_proba():
    model = LogisticRegression(max_iter=200)
    X = np.array([[1, 2], [1, -1], [2, 1], [2, 2], [3, -1]])
    y = np.array([1, 0, 1, 1, 0])
    model.fit(X, y)
    log_proba = model.predict_log_proba(X)
    assert log_proba.shape == (5,)
    assert np.all(log_proba <= 0)

def test_predict():
    model = LogisticRegression(max_iter=200)
    X = np.array([[1, 2], [1, -1], [2, 1], [2, 2], [3, -1]])
    y = np.array([1, 0, 1, 1, 0])
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (5,)
    assert np.all(np.isin(predictions, [0, 1]))

def test_score():
    model = LogisticRegression(max_iter=200)
    X = np.array([[1, 2], [1, -1], [2, 1], [2, 2], [3, -1]])
    y = np.array([1, 0, 1, 1, 0])
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.7