import pytest
import numpy as np
from miniml.sklearn.metrics._classification import accuracy_score

def test_accuracy_score():
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    expected = 0.75
    result = accuracy_score(y_true, y_pred)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"
    print("Accuracy score test passed.")

def test_accuracy_score_targets_invalid_shape():
    # Test with shape mismatch
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 1, 0])
    
    try:
        accuracy_score(y_true, y_pred)
    except ValueError as e:
        assert str(e) == "Shape mismatch: y_true shape (4,), y_pred shape (5,)"
        print("Check classification targets (invalid shape) test passed.")

def test_accuracy_score_targets_invalid_type():
    # Test with non-integer types in y_true
    y_true = np.array([1.0, 0.5, 1, 1])
    y_pred = np.array([1, 0, 1, 1])
    
    try:
        accuracy_score(y_true, y_pred)
    except ValueError as e:
        assert str(e) == "Both y_true and y_pred must be integers representing class labels."
        print("Check classification targets (invalid type) test passed.")

if __name__ == "__main__":
    pytest.main()