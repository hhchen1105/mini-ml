import numpy as np
from miniml.sklearn.naive_bayes import ComplementNB


def test_ComplementNB_basic_fit_predict():

    X = np.array([
        [3, 0],
        [4, 0],
        [0, 3],
        [0, 4],
    ])
    y = np.array([0, 0, 1, 1])

    clf = ComplementNB(alpha=1.0)
    clf.fit(X, y)

    X_test = np.array([
        [5, 0],  
        [0, 5],  
    ])

    y_pred = clf.predict(X_test)

    assert y_pred[0] == 0
    assert y_pred[1] == 1


def test_ComplementNB_predict_proba_shape_and_sum():
    X = np.array([[1, 2], [2, 1], [3, 0]])
    y = np.array([0, 1, 0])

    clf = ComplementNB()
    clf.fit(X, y)

    proba = clf.predict_proba(X)


    assert proba.shape == (3, 2)


    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_ComplementNB_negative_feature_raises():
    X = np.array([[1, -1], [0, 2]])
    y = np.array([0, 1])

    clf = ComplementNB()
    try:
        clf.fit(X, y)

        assert False, "ComplementNB should raise ValueError on negative features"
    except ValueError as e:
        assert "non-negative" in str(e)
