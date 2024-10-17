import pytest
import numpy as np
from miniml.sklearn.metrics._regression import r2_score,mean_absolute_error,mean_squared_error

def test_r2_score():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    expected = 0.9486081370449679
    result = r2_score(y_true, y_pred)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"
    print("R2 score test passed.")

def test_r2_score_targets_invalid_type():
    # Test with non-numeric type in y_true
    y_true = np.array(["a", -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    try:
        r2_score(y_true, y_pred)
    except ValueError as e:
        assert str(e) == "Both y_true and y_pred must be numeric."
        print("Check regression targets (invalid type) test passed.")

def test_r2_score_targets_invalid_shape():
    # Test with shape mismatch
    y_true = np.array([3.0, -0.5, 2.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    try:
        r2_score(y_true, y_pred)
    except ValueError as e:
        assert str(e) == "Shape mismatch: y_true shape (3,), y_pred shape (4,)"
        print("Check regression targets (invalid shape) test passed.")

def test_mean_squared_error():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    expected = 0.375
    result = mean_squared_error(y_true, y_pred)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"
    print("Mean squared error test passed.")

def test_mean_squared_error_targets_invalid_shape():
    # Test with shape mismatch
    y_true = np.array([3.0, -0.5, 2.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    try:
        mean_squared_error(y_true, y_pred)
    except ValueError as e:
        assert str(e) == "Shape mismatch: y_true shape (3,), y_pred shape (4,)"
        print("Check regression targets (invalid shape) test passed.")
    
def test_mean_squared_error_targets_invalid_type():
    # Test with non-numeric type in y_true
    y_true = np.array(["a", -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    try:
        mean_squared_error(y_true, y_pred)
    except ValueError as e:
        assert str(e) == "Both y_true and y_pred must be numeric."
        print("Check regression targets (invalid type) test passed.")

def test_mean_absolute_error():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    expected = 0.5
    result = mean_absolute_error(y_true, y_pred)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"
    print("Mean absolute error test passed.")


def test_mean_absolute_error_targets_invalid_shape():
    # Test with shape mismatch
    y_true = np.array([3.0, -0.5, 2.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    try:
        mean_absolute_error(y_true, y_pred)
    except ValueError as e:
        assert str(e) == "Shape mismatch: y_true shape (3,), y_pred shape (4,)"
        print("Check regression targets (invalid shape) test passed.")
    
def test_mean_absolute_error_targets_invalid_type():
    # Test with non-numeric type in y_true
    y_true = np.array(["a", -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    try:
        mean_absolute_error(y_true, y_pred)
    except ValueError as e:
        assert str(e) == "Both y_true and y_pred must be numeric."
        print("Check regression targets (invalid type) test passed.")

if __name__ == "__main__":
    pytest.main()
