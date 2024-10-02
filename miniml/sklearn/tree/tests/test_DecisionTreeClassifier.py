import pytest
from miniml.sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@pytest.fixture
def iris_data():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_decision_tree_classifier_fit(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    assert clf.tree is not None

def test_decision_tree_classifier_predict(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    assert len(predictions) == len(y_test)

def test_decision_tree_classifier_accuracy(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.7  # Assuming the classifier should have at least 70% accuracy on the Iris dataset

def test_decision_tree_classifier_overfit(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))
    assert train_accuracy > test_accuracy  # Overfitting scenario

if __name__ == "__main__":
    pytest.main()