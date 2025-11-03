import pytest
import numpy as np
from miniml.sklearn.ensemble.BaggingClassifier import BaggingClassifier

# import base estimators for testing
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# import dataset for testing
from sklearn.datasets import load_iris

@pytest.fixture
def iris_data():

    X, y = load_iris(return_X_y=True)
    return X, y


def test_bagging_classifier_fit(iris_data):
    
    X, y = iris_data
    n_est = 10

    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(), 
        n_estimators=n_est, 
        random_state=42
    )
    clf.fit(X, y)

    assert len(clf.estimators_) == n_est


def test_bagging_classifier_predict(iris_data):

    X, y = iris_data

    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(), 
        n_estimators=10, 
        random_state=42
    )
    clf.fit(X, y)
    
    predictions = clf.predict(X)
    
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1, 2})

# test when base estimator has predict_proba
def test_bagging_classifier_predict_proba_soft_vote(iris_data):

    X, y = iris_data
    n_classes = len(np.unique(y))
    
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=10, 
        random_state=42
    )

    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    
    assert proba.shape == (len(y), n_classes)
    assert np.all((proba >= 0) & (proba <= 1))


# test when base estimator does not have predict_proba
def test_bagging_classifier_predict_proba_hard_vote(iris_data):

    X, y = iris_data
    n_classes = len(np.unique(y))

    clf = BaggingClassifier(
        estimator=SVC(probability=False), 
        n_estimators=10, 
        random_state=42
    )
    
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    assert proba.shape == (len(y), n_classes)
    assert np.all((proba >= 0) & (proba <= 1))
