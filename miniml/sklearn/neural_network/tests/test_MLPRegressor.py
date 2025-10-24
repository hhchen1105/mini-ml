import numpy as np
import pytest
from numpy.testing import assert_allclose
from miniml.sklearn.neural_network import MLPRegressor


@pytest.mark.parametrize(
    "hidden_layer_sizes, max_iter",
    [
        ((10,), 1000),
        ((5, 5), 1500),
    ],
)
def test_MLPRegressor_fit_predict_score(hidden_layer_sizes, max_iter):
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=float)
    y = np.dot(X, np.array([1.0, 2.0])) + 3.0   # y = 1*x1 + 2*x2 + 3

    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                       max_iter=max_iter)

    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

    assert X.shape == (4, 2)

    assert hasattr(model, "coefs_")
    assert hasattr(model, "intercepts_")
