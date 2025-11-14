import numpy as np

class RadiusNeighborsRegressor:
    def __init__(self, radius=1.0):
        self.radius = float(radius)
        self.X_ = None
        self.y_ = None

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def _euclidean_distances(self, X):
        diff = X[:, None, :] - self.X_[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def predict(self, X):
        if self.X_ is None:
            raise ValueError("Not fitted yet.")
        X = np.asarray(X, dtype=float)
        dists = self._euclidean_distances(X)
        preds = []
        for i in range(X.shape[0]):
            dist_i = dists[i]
            idx = np.where(dist_i < self.radius)[0]
            if idx.size == 0:
                # 沒鄰居就拿最近的那個 y
                nearest = np.argmin(dist_i)
                preds.append(self.y_[nearest])
            else:
                # 有鄰居就平均
                preds.append(np.mean(self.y_[idx]))
        return np.array(preds)
