from miniml.sklearn.neighbors import RadiusNeighborsClassifier
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_fit_stores_training_data_and_classes():
	X_train = np.array([[0, 0], [1, 1], [2, 2]])
	y_train = np.array([0, 1, 1])
	clf = RadiusNeighborsClassifier(radius=1.5, weights="uniform").fit(X_train, y_train)

	assert_allclose(clf.X_train, X_train)
	assert_allclose(clf.y_train, y_train)
	assert_allclose(clf.classes_, np.array([0, 1]))


def test_predict_uniform_weights():
	X_train = np.array([[0, 0], [0, 1], [2, 2]])
	y_train = np.array([0, 0, 1])
	clf = RadiusNeighborsClassifier(radius=1.1, weights="uniform").fit(X_train, y_train)

	preds = clf.predict(np.array([[0.1, 0]]))
	assert_allclose(preds, np.array([0]))


def test_predict_distance_weights_prefers_close_points():
	X_train = np.array([[0, 0], [0.1, 0], [0, 2], [0, 2.5]])
	y_train = np.array([1, 1, 0, 0])
	clf = RadiusNeighborsClassifier(radius=3.0, weights="distance").fit(X_train, y_train)

	preds = clf.predict(np.array([[0, 0.05]]))
	assert_allclose(preds, np.array([1]))

	prob = clf.predict_proba(np.array([[0, 0.05]]))
	assert prob[0, 1] > prob[0, 0]


def test_predict_proba_uniform_counts():
	X_train = np.array([[0, 0], [0, 1], [1, 0], [3, 3]])
	y_train = np.array([0, 0, 1, 1])
	clf = RadiusNeighborsClassifier(radius=1.1, weights="uniform").fit(X_train, y_train)

	prob = clf.predict_proba(np.array([[0, 0]]))
	assert_allclose(prob, np.array([[2 / 3, 1 / 3]]))


def test_predict_minkowski_p_changes_inclusion():
	X_train = np.array([[0, 0], [2, 0]])
	y_train = np.array([0, 1])
	clf = RadiusNeighborsClassifier(radius=2.5, p=1).fit(X_train, y_train)

	preds = clf.predict(np.array([[1, -0.5]]))
	assert_allclose(preds, np.array([0]))


def test_predict_outlier_label_handled():
	X_train = np.array([[0, 0], [1, 1]])
	y_train = np.array([0, 1])
	clf = RadiusNeighborsClassifier(radius=0.4, outlier_label=-1).fit(X_train, y_train)

	preds = clf.predict(np.array([[2, 2]]))
	assert_allclose(preds, np.array([-1]))

	prob = clf.predict_proba(np.array([[2, 2]]))
	assert_allclose(prob, np.array([[0.0, 0.0]]))
