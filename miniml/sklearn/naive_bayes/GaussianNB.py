import numpy as np

class GaussianNB:
    def __init__(self) -> None:
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("Shape mismatch: X and y have different number of samples.")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i, :] = X_c.mean(axis=0)
            self.var_[i, :] = X_c.var(axis=0) + 1e-9
            self.class_prior_[i] = X_c.shape[0] / X.shape[0]

        return self

    def _gaussian_pdf(self, X, mean, var):
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(-((X - mean) ** 2) / (2.0 * var))
        return coeff * exponent

    def predict_proba(self, X):
        X = np.asarray(X)
        probs = []
        for i, c in enumerate(self.classes_):
            likelihood = self._gaussian_pdf(X, self.theta_[i], self.var_[i])
            joint_likelihood = np.prod(likelihood, axis=1)
            probs.append(self.class_prior_[i] * joint_likelihood)
        probs = np.array(probs).T
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
