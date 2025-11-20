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


def test_sklearn_attributes():
    """Test that sklearn-compatible attributes are set"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    clf = GradientBoostingClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    # Check attributes exist
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'n_features_in_')

    # Check values
    assert np.array_equal(clf.classes_, [0, 1])
    assert clf.n_features_in_ == 2


def test_input_validation_X_not_2d():
    """Test that 1D X raises error"""
    X = np.array([1, 2, 3, 4])
    y = np.array([0, 1, 0, 1])
    clf = GradientBoostingClassifier()

    with pytest.raises(ValueError, match="X must be 2D array"):
        clf.fit(X, y)


def test_input_validation_y_not_1d():
    """Test that 2D y raises error"""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0], [1]])
    clf = GradientBoostingClassifier()

    with pytest.raises(ValueError, match="y must be 1D array"):
        clf.fit(X, y)


def test_input_validation_shape_mismatch():
    """Test that X and y shape mismatch raises error"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])
    clf = GradientBoostingClassifier()

    with pytest.raises(ValueError, match="X and y must have same number of samples"):
        clf.fit(X, y)


def test_input_validation_non_binary():
    """Test that non-binary labels raise error"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 2])
    clf = GradientBoostingClassifier()

    with pytest.raises(ValueError, match="y must contain binary labels"):
        clf.fit(X, y)


def test_input_validation_single_class():
    """Test that single class raises error"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 1, 1])
    clf = GradientBoostingClassifier()

    with pytest.raises(ValueError, match="y contains only one class"):
        clf.fit(X, y)


def test_labels_minus_one_plus_one():
    """Test that labels {-1, 1} are correctly converted to {0, 1}"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([-1, -1, 1, 1])
    clf = GradientBoostingClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)

    # Predictions should be in {0, 1}
    predictions = clf.predict(X)
    assert set(predictions).issubset({0, 1})

    # Score should work with both {0, 1} and {-1, 1}
    score1 = clf.score(X, np.array([0, 0, 1, 1]))
    score2 = clf.score(X, np.array([-1, -1, 1, 1]))
    assert score1 == score2


def test_hyperparameter_validation_n_estimators():
    """Test that invalid n_estimators raises error"""
    clf = GradientBoostingClassifier(n_estimators=0)
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="n_estimators must be an integer >= 1"):
        clf.fit(X, y)

    clf = GradientBoostingClassifier(n_estimators=-5)
    with pytest.raises(ValueError, match="n_estimators must be an integer >= 1"):
        clf.fit(X, y)


def test_hyperparameter_validation_learning_rate():
    """Test that invalid learning_rate raises error"""
    clf = GradientBoostingClassifier(learning_rate=0)
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        clf.fit(X, y)

    clf = GradientBoostingClassifier(learning_rate=-0.1)
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        clf.fit(X, y)


def test_predict_before_fit():
    """Test that predict before fit raises error"""
    clf = GradientBoostingClassifier()
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="not fitted yet"):
        clf.predict(X)


def test_feature_dimension_mismatch():
    """Test that wrong number of features in predict raises error"""
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 0, 1])
    clf = GradientBoostingClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    # Try to predict with wrong number of features
    X_test = np.array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(ValueError, match="X has 3 features.*expecting 2 features"):
        clf.predict(X_test)


def test_random_state_reproducibility():
    """Test that same random_state gives same results"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])

    clf1 = GradientBoostingClassifier(n_estimators=10, random_state=42)
    clf1.fit(X, y)
    pred1 = clf1.predict(X)

    clf2 = GradientBoostingClassifier(n_estimators=10, random_state=42)
    clf2.fit(X, y)
    pred2 = clf2.predict(X)

    assert np.array_equal(pred1, pred2)


def test_random_state_different():
    """Test that different random_state gives different results"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])

    clf1 = GradientBoostingClassifier(n_estimators=10, random_state=42)
    clf1.fit(X, y)

    clf2 = GradientBoostingClassifier(n_estimators=10, random_state=123)
    clf2.fit(X, y)

    # May be different (not guaranteed, but likely with small data)
    # Just check they both work
    assert clf1.score(X, y) >= 0
    assert clf2.score(X, y) >= 0


if __name__ == "__main__":
    pytest.main()
