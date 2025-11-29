import numpy as np
from miniml.sklearn.base import BaseEstimator


class ExtraTreeClassifier(BaseEstimator):
    """極度隨機樹分類器 (Extra-Tree Classifier) 的簡化實作。

    在每個節點：
    1. 隨機抽樣若干特徵 (由 max_features 控制，預設 'sqrt')。
    2. 從該特徵的相鄰唯一值中點集合中隨機抽一個 threshold。
    3. 選擇造成加權 Gini 不純度最小的分割。

    參數:
    ----------
    max_depth : int | None, default=None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    max_features : int | str | None, default='sqrt'
    random_state : int | None, default=None
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.tree_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.is_fitted_ = False
        self._last_n_tries = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different number of samples.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >=2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >=1")

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        y_enc = np.searchsorted(self.classes_, y)
        self.tree_ = self._build_tree(X, y_enc, depth=0)
        self.is_fitted_ = True
        return self

    def _build_tree(self, X, y, depth):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or (n_samples < self.min_samples_split)
            or (n_samples < self.min_samples_leaf * 2)
            or (n_labels == 1)
        ):
            return {"leaf": True, "value": self._calculate_leaf_value(y)}

        feat_idx, threshold = self._find_random_split(X, y, n_samples)
        if feat_idx is None:
            return {"leaf": True, "value": self._calculate_leaf_value(y)}

        left_mask = X[:, feat_idx] < threshold
        right_mask = ~left_mask
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return {"leaf": True, "value": self._calculate_leaf_value(y)}
        return {
            "leaf": False,
            "feature": feat_idx,
            "threshold": threshold,
            "left": self._build_tree(X_left, y_left, depth + 1),
            "right": self._build_tree(X_right, y_right, depth + 1),
        }

    def _find_random_split(self, X, y, n_samples):
        best_gini = 1.0
        best = (None, None)
        if self.max_features is None:
            n_try = self.n_features_
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                n_try = max(1, int(np.sqrt(self.n_features_)))
            else:
                raise ValueError("Unsupported max_features string.")
        elif isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features int must be >=1")
            n_try = min(self.max_features, self.n_features_)
        else:
            raise ValueError("max_features must be int | str | None")
        self._last_n_tries = n_try
        feats = self.rng.choice(self.n_features_, n_try, replace=False)
        for f in feats:
            col = X[:, f]
            uniq = np.unique(col)
            if uniq.size <= 1:
                continue
            mids = (uniq[:-1] + uniq[1:]) / 2
            thr = self.rng.choice(mids)
            left = col < thr
            y_left = y[left]
            y_right = y[~left]
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            g = (len(y_left) * self._gini(y_left) + len(y_right) * self._gini(y_right)) / n_samples
            if g < best_gini:
                best_gini = g
                best = (f, thr)
        return best

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, c = np.unique(y, return_counts=True)
        p = c / len(y)
        return 1 - np.sum(p ** 2)

    def _calculate_leaf_value(self, y):
        if len(y) == 0:
            return np.ones(self.n_classes_) / self.n_classes_
        counts = np.bincount(y, minlength=self.n_classes_)
        return counts / len(y)

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Estimator not fitted.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_:
            raise ValueError("Feature mismatch.")
        return np.array([self._traverse_tree(row, self.tree_) for row in X])

    def _traverse_tree(self, sample, node):
        if node["leaf"]:
            idx = int(np.argmax(node["value"]))
            return self.classes_[idx]
        if sample[node["feature"]] < node["threshold"]:
            return self._traverse_tree(sample, node["left"])
        return self._traverse_tree(sample, node["right"])

    def predict_proba(self, X):
        if not self.is_fitted_ or self.tree_ is None:
            raise ValueError("Estimator not fitted.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_:
            raise ValueError("Feature mismatch.")
        return np.vstack([self._traverse_tree_proba(row, self.tree_) for row in X])

    def _traverse_tree_proba(self, sample, node):
        if node["leaf"]:
            return np.asarray(node["value"], dtype=float)
        if sample[node["feature"]] < node["threshold"]:
            return self._traverse_tree_proba(sample, node["left"])
        return self._traverse_tree_proba(sample, node["right"])
