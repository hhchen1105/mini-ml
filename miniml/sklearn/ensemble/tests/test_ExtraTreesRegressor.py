import pytest
import numpy as np
from miniml.sklearn.ensemble.ExtraTreesRegressor import ExtraTreesRegressor

def test_extra_trees_regressor_fit():
    # Check if model trains correct number of trees
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = ExtraTreesRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    assert len(model.trees) == 10

def test_extra_trees_regressor_predict_shape():
    # Check output shape matches target
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = ExtraTreesRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape

def test_extra_trees_regressor_deterministic_with_same_seed():
    # Same seed should give identical results
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model1 = ExtraTreesRegressor(n_estimators=5, max_depth=3, random_state=123)
    model2 = ExtraTreesRegressor(n_estimators=5, max_depth=3, random_state=123)
    model1.fit(X, y)
    model2.fit(X, y)
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    assert np.allclose(pred1, pred2)

def test_extra_trees_regressor_variance_with_different_seed():
    # Different seeds should produce different results
    X = np.array([[1, 5], [2, 4], [3, 6], [4, 5], [5, 7]])
    y = np.array([1, 2, 3, 4, 5])
    model1 = ExtraTreesRegressor(n_estimators=50, max_depth=3, max_features=1,
                                 bootstrap=True, random_state=1)
    model2 = ExtraTreesRegressor(n_estimators=50, max_depth=3, max_features=1,
                                 bootstrap=True, random_state=2)
    model1.fit(X, y)
    model2.fit(X, y)
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    assert not np.allclose(pred1, pred2)

def test_extra_trees_regressor_overfit_small_data():
    # Deep trees should overfit small data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([10, 20, 30, 40, 50])
    model = ExtraTreesRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-1)

def test_extra_trees_regressor_underfit_simple_model():
    # Very shallow model should underfit
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([10, 20, 30, 40, 50])
    model = ExtraTreesRegressor(n_estimators=1, max_depth=1, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.allclose(predictions, y, atol=1e-1)

if __name__ == "__main__":
    pytest.main()
