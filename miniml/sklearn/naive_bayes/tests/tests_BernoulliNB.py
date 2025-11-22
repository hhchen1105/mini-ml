import pytest
import numpy as np
from miniml.sklearn.naive_bayes.BernoulliNB import BernoulliNB


@pytest.fixture
def sample_data():
    X = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
    ])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_BernoulliNB_initialization():
    bnb = BernoulliNB(alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True, class_prior=[0,3, 0,2, 0,5])
    assert bnb.alpha == 1
    assert bnb.force_alpha == True
    assert bnb,binarize == 0.0
    assert bnb.fit_prior == True
    assert bnb.class_prior == [0,3, 0,2, 0,5]


def test_fit1(sample_data):
    X, y = sample_data
    bnb = BernoulliNB(alpha=1.0)
    bnb.fit(X, y)

    assert hasattr(bnb, "classes_")
    assert hasattr(bnb, "class_count_")
    assert hasattr(bnb, "feature_count_")
    assert len(bnb.classes_) == 2
    assert bnb.feature_count_.shape == (len(bnb.classes_), len(X[0]))
    assert np.all(bnb.class_count_ > 0)
    assert np.sum(bnb.class_count_) == len(X)
    assert np.isclose(np.sum(np.exp(bnb.class_log_prior_)), 1.0, atol=1e-6)


def test_fit2(sample_data):
    X, y = sample_data
    prior = [0.7, 0.3]
    bnb = BernoulliNB(alpha=1.0, class_prior=prior, fit_prior=False)
    bnb.fit(X, y)

    expect_log_prior = np.log(prior)
    np.testing.assert_allclose(bnb.class_log_prior_, expect_log_prior)


def test_fit3(sample_data):
    # 測試參數 fit_prior
    X, y = sample_data
    bnb = BernoulliNB(alpha=1.0, fit_prior=False)
    bnb.fit(X, y)

    np.testing.assert_allclose(
        np.exp(bnb.class_log_prior_), [0.5, 0.5], atol=1e-6
    )

def test_predict(sample_data):
    X, y = sample_data
    bnb = BernoulliNB()
    bnb.fit(X, y)

    y_pred = bnb.predict(X)
    assert y_pred.shape == y.shape
    assert np.all(np.isin(y_pred, bnb.classes_))


def test_predict_proba(sample_data):
    X, y = sample_data
    bnb = BernoulliNB()
    bnb.fit(X, y)

    proba = bnb.predict_proba(X)
    assert proba.shape == (len(X), len(bnb.classes_))

    np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main()
