import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        
        # Randomly initialize the centroids
        initial_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.cluster_centers_ = X[initial_indices]

        for i in range(self.max_iter):
            # Assign labels based on closest center
            self.labels_ = self._assign_labels(X)
            
            # Calculate new centers
            new_centers = self._calculate_centers(X)
            
            # Check for convergence
            if np.all(np.linalg.norm(new_centers - self.cluster_centers_, axis=1) < self.tol):
                break
            
            self.cluster_centers_ = new_centers
        
        self.inertia_ = self._calculate_inertia(X)

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_centers(self, X):
        new_centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            members = X[self.labels_ == k]
            if len(members) > 0:
                new_centers[k] = members.mean(axis=0)
        return new_centers

    def _calculate_inertia(self, X):
        distances = np.linalg.norm(X - self.cluster_centers_[self.labels_], axis=1)
        return np.sum(distances ** 2)

    def predict(self, X):
        return self._assign_labels(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_