import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        index, value, left_tree, right_tree = tree
        if x[index] <= value:
            return self._predict_sample(x, left_tree)
        else:
            return self._predict_sample(x, right_tree)

    def _mse(self, y):
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _split(self, X, y, index, value):
        left_mask = X[:, index] <= value
        right_mask = X[:, index] > value
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        best_index, best_value, best_score, best_splits = None, None, float('inf'), None
        for index in range(X.shape[1]):
            for value in np.unique(X[:, index]):
                X_left, X_right, y_left, y_right = self._split(X, y, index, value)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                score = (self._mse(y_left) * len(y_left) + self._mse(y_right) * len(y_right)) / len(y)
                if score < best_score:
                    best_index, best_value, best_score, best_splits = index, value, score, (X_left, X_right, y_left, y_right)
        return best_index, best_value, best_splits

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)
        index, value, splits = self._best_split(X, y)
        if index is None:
            return np.mean(y)
        left_tree = self._build_tree(splits[0], splits[2], depth + 1)
        right_tree = self._build_tree(splits[1], splits[3], depth + 1)
        return (index, value, left_tree, right_tree)
