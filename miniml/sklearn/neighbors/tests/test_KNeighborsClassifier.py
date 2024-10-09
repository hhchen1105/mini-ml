from miniml.sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numpy.testing import assert_allclose


def test_knn_classifier_predict():
    X_train = np.array([[1, 2], [2, 2.2], [3, 4]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[1, 2], [2, 2]])
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    expected_predictions = np.array([0, 1])
    assert_allclose(predictions, expected_predictions)


def test_knn_classifier_with_different_k():
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[2, 3]])
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    expected_prediction = np.array([0])
    assert_allclose(prediction, expected_prediction)
