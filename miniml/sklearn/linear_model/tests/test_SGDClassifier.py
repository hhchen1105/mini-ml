import pytest
import numpy as np
from miniml.sklearn.linear_model import SGDClassifier

def test_fit():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = SGDClassifier(learning_rate=0.1, n_iter=1000, alpha=0.0001)
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None


def test_predict():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = SGDClassifier(learning_rate=0.1, n_iter=1000, alpha=0.0001)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions is not None
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_predict_proba():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = SGDClassifier(learning_rate=0.1, n_iter=1000, alpha=0.0001)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(y), 2)  # 二元分類
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)


def test_score_high_accuracy():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = SGDClassifier(learning_rate=0.1, n_iter=2000, alpha=0.0001)
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.9  # 應該能很好地分開


def test_underfitting():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = SGDClassifier(learning_rate=0.01, n_iter=1, alpha=0.0001)
    model.fit(X, y)
    score = model.score(X, y)
    assert score < 0.8  # 太少迭代可能表現差


def test_overfitting():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = SGDClassifier(learning_rate=0.1, n_iter=10000, alpha=0.0001)
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.99  # 迭代足夠應該幾乎完美分類


if __name__ == "__main__":
    pytest.main()
