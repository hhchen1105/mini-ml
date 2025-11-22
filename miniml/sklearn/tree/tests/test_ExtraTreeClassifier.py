import numpy as np
import pytest
from miniml.sklearn.tree import ExtraTreeClassifier


@pytest.fixture
def iris_data():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=0.0, scale=1.0, size=(50, 4)),  # 類別 0
        rng.normal(loc=3.0, scale=1.0, size=(50, 4)),  # 類別 1
        rng.normal(loc=6.0, scale=1.0, size=(50, 4)),  # 類別 2
    ])
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)

    # 自製 train/test split
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    test_size = int(0.2 * n_samples)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
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


def test_repeatability_with_same_random_state(iris_data):
    # 同一個 random_state，重複 fit，預測/機率應一致（穩定性）
    X_train, X_test, y_train, _ = iris_data

    clf1 = ExtraTreeClassifier(random_state=123, max_depth=10)
    clf1.fit(X_train, y_train)
    y_pred1 = clf1.predict(X_test)
    proba1 = clf1.predict_proba(X_test)

    clf2 = ExtraTreeClassifier(random_state=123, max_depth=10)
    clf2.fit(X_train, y_train)
    y_pred2 = clf2.predict(X_test)
    proba2 = clf2.predict_proba(X_test)

    assert np.array_equal(y_pred1, y_pred2)
    assert np.allclose(proba1, proba2)


def test_multiple_calls_return_same_results(iris_data):
    # 同一個模型、同一筆資料，多次呼叫 predict/predict_proba 應得到相同結果
    X_train, X_test, y_train, _ = iris_data
    clf = ExtraTreeClassifier(random_state=7, max_depth=8)
    clf.fit(X_train, y_train)

    y_pred_a = clf.predict(X_test)
    y_pred_b = clf.predict(X_test)
    assert np.array_equal(y_pred_a, y_pred_b)

    proba_a = clf.predict_proba(X_test)
    proba_b = clf.predict_proba(X_test)
    assert np.allclose(proba_a, proba_b)


def test_different_random_states_can_change_predictions(iris_data):
    # 不同 random_state 應可能導致不同的模型與預測（顯示隨機性影響）
    X_train, X_test, y_train, _ = iris_data

    seeds = [0, 1, 2, 3, 4]
    preds = []
    probas = []
    for s in seeds:
        clf = ExtraTreeClassifier(random_state=s, max_depth=2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        preds.append(tuple(y_pred.tolist()))
        proba = clf.predict_proba(X_test)
        # 將機率攤平成一維並四捨五入，避免浮點微小差異造成不必要的唯一化
        probas.append(tuple(np.round(proba.ravel(), 6).tolist()))

    unique_pred_count = len(set(preds))
    unique_proba_count = len(set(probas))

    # 允許其中一者顯示差異即可，降低測試易碎性
    assert unique_pred_count >= 2 or unique_proba_count >= 2


def test_max_features_controls_try_count():
    # 使用 5 維人工資料，檢查 _last_n_tries 反映 max_features 設定
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # max_features=None -> 嘗試全部特徵
    clf_all = ExtraTreeClassifier(random_state=0, max_features=None, max_depth=2)
    clf_all.fit(X, y)
    # 觸發一次 split，使 _last_n_tries 記錄本節點嘗試數
    _ = clf_all.predict(X[:3])
    assert clf_all._last_n_tries == 5

    # max_features='sqrt' -> floor(sqrt(5))=2
    clf_sqrt = ExtraTreeClassifier(random_state=0, max_features='sqrt', max_depth=2)
    clf_sqrt.fit(X, y)
    _ = clf_sqrt.predict(X[:3])
    assert clf_sqrt._last_n_tries == 2

    # max_features=3 -> 至多 3
    clf_int = ExtraTreeClassifier(random_state=0, max_features=3, max_depth=2)
    clf_int.fit(X, y)
    _ = clf_int.predict(X[:3])
    assert clf_int._last_n_tries == 3
