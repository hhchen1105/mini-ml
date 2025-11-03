import pytest
import numpy as np
from miniml.sklearn.ensemble.GradientBoostingClassifier import GradientBoostingClassifier


def test_gradient_boosting_classifier_fit():
    """Test that fit method trains the correct number of trees"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = GradientBoostingClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    assert len(clf.trees) == 10
    assert clf.init_prediction is not None


def test_gradient_boosting_classifier_predict():
    """Test that predict method returns valid class labels"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_gradient_boosting_classifier_predict_proba():
    """Test that predict_proba returns valid probabilities"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_gradient_boosting_classifier_score():
    """Test that score method returns a value between 0 and 1"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert 0 <= score <= 1


def test_gradient_boosting_classifier_learning_rate():
    """Test that different learning rates affect predictions"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf1 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, random_state=42)
    clf1.fit(X, y)
    score1 = clf1.score(X, y)

    clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, random_state=42)
    clf2.fit(X, y)
    score2 = clf2.score(X, y)

    # Both should achieve reasonable accuracy
    assert score1 >= 0.5
    assert score2 >= 0.5


def test_gradient_boosting_classifier_separable_data():
    """Test on linearly separable data"""
    # Create a simple linearly separable dataset
    np.random.seed(42)
    X_class0 = np.random.randn(20, 2) - 2
    X_class1 = np.random.randn(20, 2) + 2
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * 20 + [1] * 20)

    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    clf.fit(X, y)
    score = clf.score(X, y)

    # Should achieve high accuracy on separable data
    assert score >= 0.85


def test_gradient_boosting_classifier_max_depth():
    """Test that max_depth parameter is respected"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf1 = GradientBoostingClassifier(n_estimators=10, max_depth=1, random_state=42)
    clf1.fit(X, y)

    clf2 = GradientBoostingClassifier(n_estimators=10, max_depth=5, random_state=42)
    clf2.fit(X, y)

    # Both should train without errors
    assert len(clf1.trees) == 10
    assert len(clf2.trees) == 10


@pytest.mark.parametrize(
    "n_estimators, learning_rate, max_depth",
    [
        (10, 0.1, 3),
        (50, 0.05, 2),
        (20, 0.2, 4),
    ],
)
def test_gradient_boosting_classifier_parameters(n_estimators, learning_rate, max_depth):
    """Test with different parameter combinations"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    clf.fit(X, y)
    predictions = clf.predict(X)
    proba = clf.predict_proba(X)
    score = clf.score(X, y)

    assert len(predictions) == len(y)
    assert proba.shape == (len(y), 2)
    assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main()
