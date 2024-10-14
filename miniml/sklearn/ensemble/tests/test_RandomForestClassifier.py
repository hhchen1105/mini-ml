import pytest
import numpy as np
from miniml.sklearn.ensemble.RandomForestClassifier import RandomForestClassifier

def test_random_forest_classifier_fit():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    assert len(clf.trees) == 10
    assert len(clf.feature_indices) == 10

def test_random_forest_classifier_predict():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})

def test_random_forest_classifier_predict_proba():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1)

def test_random_forest_classifier_score():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert 0 <= score <= 1

if __name__ == "__main__":
    pytest.main()