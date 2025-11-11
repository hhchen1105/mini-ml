from miniml.sklearn.neighbors import KNeighborsRegressor
import numpy as np
from numpy.testing import assert_allclose


def test_knn_regressor_predict():
    X_train = np.array([[1, 2], [2, 2.2], [3, 4]])
    y_train = np.array([10, 20, 30])
    X_test = np.array([[1, 2], [2, 2]])
    
    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    expected_predictions = np.array([10, 20])    
    assert_allclose(predictions, expected_predictions)


def test_knn_regressor_with_different_k():
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([10, 20, 30, 40])
    X_test = np.array([[2, 3]])
    
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    
    expected_prediction = np.array([20])    
    assert_allclose(prediction, expected_prediction)