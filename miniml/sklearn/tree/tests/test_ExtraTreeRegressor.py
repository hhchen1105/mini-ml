import numpy as np
import pytest
from miniml.sklearn.tree.ExtraTreeRegressor import ExtraTreeRegressor


def test_fit_and_predict_mean_model():
    # 準備訓練資料
    X = np.array([[1], [2], [3], [4]])
    y = np.array([10.0, 20.0, 30.0, 40.0])

    # 建立並訓練模型
    model = ExtraTreeRegressor()
    model.fit(X, y)

    # 用新資料做預測
    X_test = np.array([[5], [6]])
    preds = model.predict(X_test)

    # 模型應該預測的是訓練 y 的平均值
    expected = np.mean(y)

    assert preds.shape == (2,)
    assert np.allclose(preds, expected)


def test_predict_before_fit_raises_error():
    model = ExtraTreeRegressor()
    X_test = np.array([[1], [2]])

    # 還沒 fit 就 predict → 應該會丟 RuntimeError
    with pytest.raises(RuntimeError):
        model.predict(X_test)
