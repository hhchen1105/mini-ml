import numpy as np
import pytest
from miniml.sklearn.tree import ExtraTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


@pytest.fixture
def iris_data():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_extra_tree_classifier_fit_flags(iris_data):
    X_train, _, y_train, _ = iris_data
    clf = ExtraTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    assert clf.is_fitted_ is True
    assert clf.tree_ is not None
    assert clf.n_features_ == X_train.shape[1]
    assert set(clf.classes_) == set(np.unique(y_train))


def test_predict_proba_shape_and_normalization(iris_data):
    X_train, X_test, y_train, _ = iris_data
    clf = ExtraTreeClassifier(random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (X_test.shape[0], clf.n_classes_)
    row_sums = np.sum(proba, axis=1)
    assert np.allclose(row_sums, 1.0)
    assert np.min(proba) >= 0.0 and np.max(proba) <= 1.0


def test_predict_matches_proba_argmax(iris_data):
    X_train, X_test, y_train, _ = iris_data
    clf = ExtraTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    argmax_labels = clf.classes_[np.argmax(proba, axis=1)]
    assert np.array_equal(y_pred, argmax_labels)


def test_min_samples_leaf_prevents_split_and_returns_class_distribution():
    # 構造一個小資料集，使得 min_samples_leaf 的限制阻止任何分割
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1])  # 類別分佈: 0 -> 3/4, 1 -> 1/4
    clf = ExtraTreeClassifier(min_samples_leaf=3, random_state=0)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    # 每一列都應該等於整體的類別分佈
    expected = np.array([0.75, 0.25])  # 對應於 classes_ = [0, 1]
    assert proba.shape == (4, 2)
    assert np.allclose(proba, np.tile(expected, (4, 1)))
    # predict 應為機率最大的類別 0
    y_pred = clf.predict(X)
    assert np.array_equal(y_pred, np.zeros_like(y))


def test_errors_before_fit_and_feature_mismatch():
    clf = ExtraTreeClassifier()
    with pytest.raises(ValueError):
        clf.predict_proba(np.array([[0.0, 1.0]]))

    X = np.array([[0.0, 1.0], [1.0, 2.0]])
    y = np.array([0, 1])
    clf.fit(X, y)
    with pytest.raises(ValueError):
        # 特徵數不匹配
        clf.predict_proba(np.array([[0.5]]))


def test_predict_accepts_1d_sample():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([0, 0, 1, 1])
    clf = ExtraTreeClassifier(random_state=42)
    clf.fit(X, y)
    pred = clf.predict(np.array([1.5, 2.5]))  # 1D 單一樣本
    assert pred.shape == (1,)
