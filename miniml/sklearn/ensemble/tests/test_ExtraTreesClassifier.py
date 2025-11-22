import pytest
import numpy as np
from miniml.sklearn.ensemble.ExtraTreesClassifier import ExtraTreesClassifier


def test_extra_trees_classifier_fit():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    clf = ExtraTreesClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    # 檢查有沒有真的建立 10 棵樹與 10 組特徵索引
    assert len(clf.trees) == 10
    assert len(clf.feature_indices) == 10


def test_extra_trees_classifier_predict():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    clf = ExtraTreesClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    predictions = clf.predict(X)

    # 預測長度跟 y 一樣
    assert len(predictions) == len(y)
    # 預測類別只會是 0 或 1（因為 y 只有 0/1）
    assert set(predictions).issubset({0, 1})


def test_extra_trees_classifier_predict_proba():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    clf = ExtraTreesClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    n_classes = len(np.unique(y))

    # 機率矩陣形狀： (樣本數, 類別數)
    assert proba.shape == (len(y), n_classes)
    # 每一列機率總和要等於 1
    assert np.allclose(proba.sum(axis=1), 1)


def test_extra_trees_classifier_score():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    clf = ExtraTreesClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    score = clf.score(X, y)

    # 準確率必須介於 0 和 1 之間
    assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main()
``