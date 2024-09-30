import pytest
import numpy as np
from miniml.sklearn.cluster.KMeans import KMeans

def test_kmeans_initialization():
    kmeans = KMeans(n_clusters=3, max_iter=100, tol=1e-3, random_state=42)
    assert kmeans.n_clusters == 3
    assert kmeans.max_iter == 100
    assert kmeans.tol == 1e-3
    assert kmeans.random_state == 42

def test_kmeans_fit():
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    assert kmeans.cluster_centers_.shape == (2, 2)
    assert kmeans.labels_.shape == (6,)
    assert kmeans.inertia_ is not None

def test_kmeans_predict():
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    assert labels.shape == (6,)
    assert np.array_equal(labels, kmeans.labels_)

def test_kmeans_fit_predict():
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X)
    assert labels.shape == (6,)
    assert np.array_equal(labels, kmeans.labels_)

def test_kmeans_convergence():
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    kmeans = KMeans(n_clusters=2, max_iter=1, tol=1e-10, random_state=42)
    kmeans.fit(X)
    assert kmeans.cluster_centers_ is not None
    assert kmeans.labels_ is not None
    assert kmeans.inertia_ is not None

def test_kmeans_convergence2():
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    kmeans1 = KMeans(n_clusters=2, max_iter=1, tol=1e-10, random_state=42)
    kmeans1.fit(X)
    kmeans2 = KMeans(n_clusters=2, max_iter=1, tol=1e-10, random_state=42)
    kmeans2.fit(X)
    assert np.array_equal(kmeans1.cluster_centers_, kmeans2.cluster_centers_)
    assert np.array_equal(kmeans1.labels_, kmeans2.labels_)
    assert kmeans1.inertia_ == kmeans2.inertia_

if __name__ == "__main__":
    pytest.main()