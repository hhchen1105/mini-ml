import numpy as np
from ..tree.DecisionTreeClassifier import DecisionTreeClassifier
from miniml.sklearn.base import BaseEstimator


class ExtraTreesClassifier(BaseEstimator):
    def __init__(
        self,
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        random_state=None,
        bootstrap=False,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.bootstrap = bootstrap

        # 每棵樹 & 每棵樹用到的特徵索引
        self.trees = []
        self.feature_indices = []

        # 統一用一個亂數產生器，確保 reproducibility
        self._rng = np.random.RandomState(self.random_state)

        # 這些在 fit 之後才會被設定
        self.classes_ = None       # 原始類別標籤
        self.n_classes_ = None     # 類別數
        self.n_features_ = None    # 特徵數（訓練時）

    # -------------------------------------------------------
    # 決定實際使用多少個特徵
    # -------------------------------------------------------
    def _get_max_features(self, n_features):
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features

    # -------------------------------------------------------
    # 一些內部檢查函式
    # -------------------------------------------------------
    def _check_is_fitted(self):
        if not self.trees:
            raise ValueError(
                "This ExtraTreesClassifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        if (
            self.n_features_ is None
            or self.n_classes_ is None
            or self.classes_ is None
        ):
            raise ValueError(
                "Model attributes are not initialized. Make sure 'fit' has been called."
            )

    def _check_input_features(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array-like, got shape {X.shape}")
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but ExtraTreesClassifier "
                f"was fitted with {self.n_features_} features."
            )
        return X

    # -------------------------------------------------------
    # ExtraTrees 的訓練邏輯
    # -------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)

        # 紀錄訓練時看到的特徵數
        self.n_features_ = n_features

        # 把原始 label 映射成 0 ~ n_classes-1
        # self.classes_：原始類別
        # y_encoded：對應的整數編號（0 ~ n_classes-1）
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        self.trees = []
        self.feature_indices = []

        for _ in range(self.n_estimators):
            # 1. 是否使用 bootstrap 抽樣
            if self.bootstrap:
                indices = self._rng.choice(n_samples, n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y_encoded[indices]
            else:
                # ExtraTrees 在 bootstrap=False 時，用全部資料但一樣保留隨機性（特徵子集）
                X_sample = X
                y_sample = y_encoded

            # 2. 隨機選特徵子集（和 RandomForest 類似）
            feature_indices = self._rng.choice(
                n_features, max_features, replace=False
            )
            self.feature_indices.append(feature_indices)

            # 3. 建一棵樹
            #    這裡用你們作業的 DecisionTreeClassifier，介面不支援 random_state / splitter
            tree = DecisionTreeClassifier(max_depth=self.max_depth)

            # 4. 用特徵子集訓練（注意使用的是 y_encoded）
            tree.fit(X_sample[:, feature_indices], y_sample)

            # 5. 存起來
            self.trees.append(tree)

        return self

    # -------------------------------------------------------
    # 預測：多數決投票
    # -------------------------------------------------------
    def predict(self, X):
        """
        對輸入 X，讓每棵樹都做預測，然後對每個樣本做多數決（voting）。
        回傳的是「原始類別」（不是 0 ~ n_classes-1）。
        """
        self._check_is_fitted()
        X = self._check_input_features(X)

        # tree_predictions 形狀：(n_estimators, n_samples)
        # 每棵樹輸出的 label 是 0 ~ n_classes-1（encoded）
        tree_predictions = np.array(
            [
                tree.predict(X[:, feature_indices])
                for tree, feature_indices in zip(self.trees, self.feature_indices)
            ]
        )

        # 對每個樣本統計各類別票數，選出最多票的類別（encoded）
        encoded_majority = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes_).argmax(),
            arr=tree_predictions,
            axis=0,
        )

        # 映射回原始類別標籤
        return self.classes_[encoded_majority]

    def predict_proba(self, X):
        """
        對每棵樹先做 predict，然後對每個樣本統計「各類別出現次數 / 樹的數量」，當作該類別的機率。
        回傳形狀：(n_samples, n_classes)，欄位順序對應 self.classes_。
        """
        self._check_is_fitted()
        X = self._check_input_features(X)

        tree_predictions = np.array(
            [
                tree.predict(X[:, feature_indices])
                for tree, feature_indices in zip(self.trees, self.feature_indices)
            ]
        )

        # 對每個樣本（axis=0）統計類別出現次數，然後除以樹的數量
        proba = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes_) / len(x),
            arr=tree_predictions,
            axis=0,
        )

        # proba 現在形狀是 (n_classes, n_samples)，轉置成 (n_samples, n_classes)
        return proba.T

    def score(self, X, y):
        """
        最簡單的準確率：預測正確的比例
        """
        X = np.asarray(X)
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)
