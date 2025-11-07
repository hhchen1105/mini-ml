import numpy as np
from ..tree.DecisionTreeClassifier import DecisionTreeClassifier


class ExtraTreesClassifier:
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

        self.trees = []             # 存每一棵 DecisionTreeClassifier
        self.feature_indices = []   # 存每棵樹使用的特徵索引

        # 單一亂數產生器：避免到處設 seed
        self._rng = np.random.RandomState(self.random_state)

    # ---------（跟你同學一樣，用來決定 max_features 實際要用幾個）---------
    def _get_max_features(self, n_features):
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features

    # --------- ExtraTrees 的訓練邏輯 ---------
    def fit(self, X, y):
       
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)

        self.trees = []
        self.feature_indices = []

        for _ in range(self.n_estimators):
            # 1. 是否使用 bootstrap 抽樣
            if self.bootstrap:
                indices = self._rng.choice(n_samples, n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                # ExtraTrees 預設用全部資料
                X_sample = X
                y_sample = y

            # 2. 隨機選特徵子集（這點和 RandomForest 一樣）
            feature_indices = self._rng.choice(
                n_features, max_features, replace=False
            )
            self.feature_indices.append(feature_indices)

            # 3. 建一棵樹
            #    這裡用的是 DecisionTreeClassifier
            #    在真正的 sklearn 內部會用 ExtraTreeClassifier 當 base estimator，
            #    但在你們 mini-ml 專案中，先沿用 DecisionTreeClassifier 應該就足夠。
            tree = DecisionTreeClassifier(max_depth=self.max_depth)

            # 4. 用特徵子集訓練
            tree.fit(X_sample[:, feature_indices], y_sample)

            # 5. 存起來
            self.trees.append(tree)

        return self

    # --------- 預測：跟 RandomForest 的 predict 幾乎一樣，做「投票」 ---------
    def predict(self, X):
        """
        對輸入 X，讓每棵樹都做預測，然後對每個樣本做多數決（voting）。
        """
        X = np.asarray(X)

        # tree_predictions 形狀：
        #   (n_estimators, n_samples)
        tree_predictions = np.array(
            [
                tree.predict(X[:, feature_indices])
                for tree, feature_indices in zip(self.trees, self.feature_indices)
            ]
        )

        # 對 axis=0（每個樣本）收集所有樹的預測，再用 bincount 找出出現次數最多的類別
        # 這裡 minlength=2 是假設至少有兩個類別（0, 1）；若你要支援多類別，
        # 可以改成根據 y 的唯一值數量來決定 minlength。
        return np.squeeze(
            np.apply_along_axis(
                lambda x: np.bincount(x, minlength=2).argmax(),
                arr=tree_predictions,
                axis=0,
            )
        )

    def predict_proba(self, X):
        """
        對每棵樹先做 predict，然後對每個樣本統計「各類別出現次數 / 樹的數量」，
        當作該類別的機率。
        回傳形狀：(n_samples, n_classes)
        """
        X = np.asarray(X)

        tree_predictions = np.array(
            [
                tree.predict(X[:, feature_indices])
                for tree, feature_indices in zip(self.trees, self.feature_indices)
            ]
        )

        # 對每個樣本（axis=0）統計類別出現次數，然後除以樹的數量
        proba = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=2) / len(x),
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
