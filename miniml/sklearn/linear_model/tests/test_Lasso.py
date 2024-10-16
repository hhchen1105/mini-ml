import pytest
import numpy as np
from miniml.sklearn.linear_model import Lasso

def test_lasso_initialization():
    model = Lasso(alpha=0.1)
    assert model.alpha == 0.1

def test_lasso_fit():
    model = Lasso(alpha=0.1)
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = [1, 2, 3]
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None

def test_lasso_predict():
    model = Lasso(alpha=0.1)
    X_train = np.array([[1, 2], [2, 3], [3, 4]])
    y_train = [1, 2, 3]
    model.fit(X_train, y_train)
    X_test = [[4, 5], [5, 6]]
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)

def test_lasso_score():
    model = Lasso(alpha=0.1)
    X_train = np.array([[1, 2], [2, 3], [3, 4]])
    y_train = [1, 2, 3]
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    assert score >= 0.8 and score <= 1