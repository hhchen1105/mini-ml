import pytest
import numpy as np
import os
from miniml.sklearn.linear_model import LinearRegression

# Create a test fixture for data
@pytest.fixture
def data():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1*x1 + 2*x2 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    return X, y

def test_linear_regression_fit_predict(data):
    """Test basic fitting and prediction functionality"""
    X, y = data
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    score = model.score(X, y)
    
    # Verify prediction shape is correct
    assert predictions.shape == y.shape
    # Verify score is 1.0 (because this is perfect linear data)
    assert score == 1.0
    # Verify predictions are very close to actual values
    np.testing.assert_array_almost_equal(predictions, y)

def test_get_and_set_params():
    """Test parameter getting and setting (get_params/set_params)"""
    model = LinearRegression()
    params = model.get_params()
    
    # Verify returned value is a dictionary
    assert isinstance(params, dict)
    
    # Test setting parameters
    new_model = LinearRegression()
    new_model.set_params(**params)
    
    assert new_model.get_params() == params