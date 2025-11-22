import pytest
import numpy as np
from miniml.sklearn.naive_bayes.CategoricalNB import CategoricalNB
from numpy.testing import assert_allclose

@pytest.mark.parametrize(
    "alpha, fit_prior",
    [
        (1.0, True, 0.5),  # With smoothing
        (1.0e-10, True, 1.0) # Minimal smoothing, should be more confident
    ],
)
def test_CategoricalNB_simple(alpha, fit_prior):
    # Simple dataset
    # Feature 0: 3 categories (0, 1, 2)
    # Feature 1: 2 categories (0, 1)
    X = np.array([
        [0, 0], 
        [0, 1], 
        [1, 0], 
        [2, 1]
    ])
    y = np.array([0, 0, 1, 1])
    
    clf = CategoricalNB(alpha=alpha, fit_prior=fit_prior)
    clf.fit(X, y)
    
    # Check basic attributes
    assert clf.n_features_in_ == 2
    assert clf.n_classes_ == 2
    # Feature 0 has max value 2 -> 3 categories
    # Feature 1 has max value 1 -> 2 categories
    assert clf.n_categories_ == [3, 2] 
    
    # Test Prediction on training data
    y_pred = clf.predict(X)
    assert y_pred.shape == (4,)
    
    # Test Predict Proba
    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (4, 2)
    assert_allclose(np.sum(y_proba, axis=1), 1.0) # Probabilities must sum to 1
    
    # Specific check: 
    test_sample = np.array([[0, 0]])
    prediction = clf.predict(test_sample)
    assert prediction[0] == 0

def test_CategoricalNB_shapes():
    rng = np.random.RandomState(42)
    X = rng.randint(5, size=(10, 3)) # 10 samples, 3 features, max cat index 4
    y = rng.randint(3, size=(10))    # 3 classes
    
    clf = CategoricalNB()
    clf.fit(X, y)
    
    assert clf.class_count_.shape == (3,)
    assert len(clf.feature_log_prob_) == 3

    # Check shape of first feature's log prob: (n_classes, n_categories)
    assert clf.feature_log_prob_[0].shape[0] == 3
    assert clf.feature_log_prob_[0].shape[1] >= 5

def test_CategoricalNB_error():
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    clf = CategoricalNB()
    clf.fit(X, y)
    
    # Test data has 1 feature，but training data has 2
    X_bad = np.array([[0], [1]])
    
    with pytest.raises((IndexError, ValueError)):
        clf.predict(X_bad)