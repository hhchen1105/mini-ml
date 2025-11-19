import numpy as np

class RadiusNeighborsRegressor:
    def __init__(self, radius=1.0, weights="uniform"):
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError(f"radius must be positive; got {radius!r}")

        if weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance'; got {weights!r}"
            )

        self.radius = radius
        self.weights = weights

        self.X_ = None
        self.y_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array; got ndim={X.ndim}")

        y = np.asarray(y, dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent numbers of samples")

        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def _euclidean_distances(self, X):
        diff = X[:, None, :] - self.X_[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def predict(self, X):
        if self.X_ is None:
            raise ValueError(
                "This RadiusNeighborsRegressor instance is not fitted yet."
            )

        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive; got {self.radius!r}")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array; got ndim={X.ndim}")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"RadiusNeighborsRegressor is expecting {self.n_features_in_} features."
            )

        dists = self._euclidean_distances(X)
        preds = []

        for i in range(X.shape[0]):
            dist_i = dists[i]
            idx = np.where(dist_i < self.radius)[0]

            if idx.size == 0:
                nearest = np.argmin(dist_i)
                preds.append(self.y_[nearest])
                continue

            neighbor_dists = dist_i[idx]
            neighbor_y = self.y_[idx]

            zero_mask = neighbor_dists == 0.0
            if np.any(zero_mask):
                same_y = neighbor_y[zero_mask]
                y_pred = same_y.mean(axis=0)
                preds.append(y_pred)
                continue

            if self.weights == "uniform":
                y_pred = neighbor_y.mean(axis=0)
            else:
                w = 1.0 / neighbor_dists
                y_pred = np.average(neighbor_y, axis=0, weights=w)

            preds.append(y_pred)

        return np.asarray(preds, dtype=float)
