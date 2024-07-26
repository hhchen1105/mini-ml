from miniml.sklearn.linear_model import LinearRegression
import numpy as np

def test_LinearRegression():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression()
    reg.fit(X, y)
    assert reg.intercept_ == 3
    print(reg.coef_)
    assert reg.coef_ == np.array([1,2])
