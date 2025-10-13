import pytest
import numpy as np
from numpy.testing import assert_allclose
from miniml.sklearn.naive_bayes import GaussianNB


@pytest.mark.parametrize(
    "X, y, expected_means, expected_vars, expected_priors",
    [
        (
            np.array([[1.0, 2.0],
                      [1.2, 2.1],
                      [3.8, 3.9],
                      [4.0, 4.1]]),
            np.array([0, 0, 1, 1]),
            np.array([[1.1, 2.05], [3.9, 4.0]]),
            np.array([[0.01, 0.0025], [0.01, 0.01]]),
            np.array([0.5, 0.5]),
        ),
    ],
)
def test_GaussianNB_fit(X, y, expected_means, expected_vars, expected_priors):
    """測試 GaussianNB.fit() 是否能正確估計各類別參數"""
    model = GaussianNB()
    model.fit(X, y)

    # test fitted parameters
    assert_allclose(model.theta_, expected_means, atol=0.05)
    assert_allclose(model.var_, expected_vars, atol=0.05)
    assert_allclose(model.class_prior_, expected_priors, atol=1e-6)

    # test that class count is correct
    assert len(model.classes_) == 2
    print("GaussianNB fit test passed.")


def test_GaussianNB_predict_and_proba():
    """測試 predict() 與 predict_proba()"""
    X = np.array([[1.0, 2.0],
                  [1.1, 2.1],
                  [3.9, 3.9],
                  [4.1, 4.0]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNB()
    model.fit(X, y)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    # 機率 shape 檢查
    assert probs.shape == (len(X), len(model.classes_))

    # 機率總和應該為 1
    assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-6)

    # 預測 shape 檢查
    assert preds.shape == y.shape

    # 預測的類別必須出現在 classes_ 中
    assert set(preds).issubset(set(model.classes_))
    print("GaussianNB predict/proba test passed.")


def test_GaussianNB_score():
    """測試 score() 應能準確評估分類正確率"""
    X = np.array([[1.0, 2.0],
                  [1.1, 2.2],
                  [3.9, 3.8],
                  [4.0, 4.1]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNB()
    model.fit(X, y)
    score = model.score(X, y)

    # 在訓練資料上應該接近完美分類
    assert 0.9 <= score <= 1.0
    print("GaussianNB score test passed.")


def test_GaussianNB_invalid_input():
    """測試輸入資料不匹配時應拋出錯誤"""
    X = np.array([[1.0, 2.0], [2.0, 3.0]])
    y = np.array([0])  # 故意給錯長度

    model = GaussianNB()
    try:
        model.fit(X, y)
    except ValueError:
        print("GaussianNB invalid input shape test passed.")
        return
    pytest.fail("GaussianNB should raise ValueError for shape mismatch.")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
