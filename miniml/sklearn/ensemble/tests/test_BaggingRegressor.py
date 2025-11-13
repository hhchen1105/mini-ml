import pytest
import numpy as np
from miniml.sklearn.ensemble.BaggingRegressor import BaggingRegressor
from miniml.sklearn.tree.DecisionTreeRegressor import DecisionTreeRegressor

@pytest.fixture
def data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    return X, y

def test_bagging_regressor_fit(data):
    X, y = data
    n_estimators = 10
    model = BaggingRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    
    # Assert: Check if 10 trees were successfully created
    assert len(model.trees) == n_estimators

def test_bagging_regressor_predict(data):
    X, y = data
    model = BaggingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Assert: Check if the prediction shape matches y's shape
    assert predictions.shape == y.shape

def test_bagging_regressor_score(data):
    X, y = data
    model = BaggingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    score = model.score(X, y)
    
    assert score <= 1.0

# Test Fix 1: Flexible base_estimator
def test_bagging_regressor_custom_estimator(data):
    X, y = data
    # Create a custom, shallow decision tree
    custom_tree = DecisionTreeRegressor(max_depth=1)
    
    model = BaggingRegressor(base_estimator=custom_tree, n_estimators=5, random_state=42)
    model.fit(X, y)
    
    # Assert: Check if the trees in the model truly have max_depth=1
    assert len(model.trees) == 5
    assert model.trees[0].max_depth == 1

# Test Fix 4: max_samples boundary test
@pytest.mark.parametrize("invalid_max_samples", [0.0, -0.1, 1.1, 2.0])
def test_bagging_regressor_max_samples_invalid(data, invalid_max_samples):
    X, y = data
    model = BaggingRegressor(max_samples=invalid_max_samples, random_state=42)
    
    # We "expect" the fit function to "raise ValueError"
    with pytest.raises(ValueError):
        model.fit(X, y)

# Test Fix 5: Reproducibility test
def test_bagging_regressor_reproducibility(data):
    X, y = data
    SEED = 42

    # Model 1 (Seed = 42)
    model_1 = BaggingRegressor(n_estimators=5, random_state=SEED)
    model_1.fit(X, y)
    preds_1 = model_1.predict(X)

    # Model 2 (Using the "same" SEED)
    model_2 = BaggingRegressor(n_estimators=5, random_state=SEED)
    model_2.fit(X, y)
    preds_2 = model_2.predict(X)

    # Model 3 (Using a "different" SEED)
    model_3 = BaggingRegressor(n_estimators=5, random_state=SEED + 1)
    model_3.fit(X, y)
    preds_3 = model_3.predict(X)

    # Assert 1: Same seed -> Same result
    assert np.array_equal(preds_1, preds_2)

    # Assert 2: Different seed -> Different result
    assert not np.array_equal(preds_1, preds_3)