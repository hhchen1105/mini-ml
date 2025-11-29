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

def test_save_and_load_model(data, tmp_path):
    """
    Test model saving and loading (save_model/load_model).
    Use tmp_path fixture to avoid creating temporary files in the project directory.
    """
    X, y = data
    model = LinearRegression()
    model.fit(X, y)
    
    original_pred = model.predict(X)
    
    save_file = tmp_path / "test_model.pkl"
    
    # 1. Save model
    model.save_model(str(save_file))
    
    # Verify file was created
    assert os.path.exists(save_file)
    
    # 2. Load model
    loaded_model = LinearRegression.load_model(str(save_file))
    
    loaded_pred = loaded_model.predict(X)
    
    # 3. Verify loaded model's prediction is identical to original model
    np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    # Verify score is also identical
    assert model.score(X, y) == loaded_model.score(X, y)
