import numpy as np

class ExtraTreeClassifier:
    """
    一個極度隨機樹分類器 (Extra-Tree Classifier) 的簡化實作。

    這個分類器在建立樹的過程中，會隨機選擇特徵和隨機選擇分割點
    來增加隨機性，這有助於減少 variance。

    參數:
    ----------
    max_depth : int, 預設=None
        樹的最大深度。如果為 None，則節點將一直擴展，直到所有葉子都是純的
        或者直到所有葉子包含的樣本數少於 min_samples_split。

    min_samples_split : int, 預設=2
        分割一個內部節點所需的最少樣本數。

    min_samples_leaf : int, 預設=1
        一個葉節點上所需的最小樣本數。

    random_state : int, 預設=None
        用於控制隨機性的種子。
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state) # 隨機數生成器
        self.tree_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.is_fitted_ = False # 在初始化時設定為 False

    def fit(self, X, y):
        """
        從訓練資料 (X, y) 建立一個 Extra-Tree 分類器。

        參數:
        ----------
        X : array-like, shape (n_samples, n_features)
            訓練輸入樣本。
        y : array-like, shape (n_samples,)
            目標類別標籤。
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # 為了方便計算葉節點的值，我們將 y 轉換為 0 到 n_classes-1 的索引
        y_encoded = np.searchsorted(self.classes_, y)
        
        self.tree_ = self._build_tree(X, y_encoded, depth=0)
        self.is_fitted_ = True # 在 fit 成功後設定為 True
        return self

    def _build_tree(self, X, y, depth):
        """遞迴建立樹"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 停止條件 (Stopping criteria)
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (n_samples < self.min_samples_leaf * 2) or \
           (n_labels == 1):
            
            # 建立葉節點
            leaf_value = self._calculate_leaf_value(y)
            return {"leaf": True, "value": leaf_value}

        # 尋找一個隨機的分割
        feature_idx, threshold = self._find_random_split(X, y, n_samples)

        # 如果找不到有效的分割
        if feature_idx is None:
            leaf_value = self._calculate_leaf_value(y)
            return {"leaf": True, "value": leaf_value}

        # 分割資料
        left_indices = X[:, feature_idx] < threshold
        right_indices = ~left_indices
        
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # 檢查是否滿足 min_samples_leaf
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            leaf_value = self._calculate_leaf_value(y)
            return {"leaf": True, "value": leaf_value}

        # 遞迴建立左右子樹
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return {
            "leaf": False,
            "feature": feature_idx,
            "threshold": threshold,
            "left": left_child,
            "right": right_child
        }

    def _find_random_split(self, X, y, n_samples):
        """
        尋找最佳的「隨機」分割點。
        這會嘗試 n_features_ 次隨機特徵和隨機閾值。
        """
        best_gini = 1.0
        best_split = (None, None)
        n_tries = self.n_features_ # 嘗試次數，可以調整 (例如 sqrt(n_features))

        # 隨機選擇要嘗試的特徵
        feature_indices = self.rng.choice(self.n_features_, n_tries, replace=True)

        for feat_idx in feature_indices:
            X_col = X[:, feat_idx]
            min_val, max_val = np.min(X_col), np.max(X_col)

            # 如果特徵值都相同，無法分割
            if min_val == max_val:
                continue

            # 隨機選擇一個閾值
            threshold = self.rng.uniform(min_val, max_val)

            # 計算 Gini 不純度
            left_indices = X_col < threshold
            y_left = y[left_indices]
            y_right = y[~left_indices]

            # 如果一邊為空，這不是一個好的分割
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gini = (len(y_left) * self._gini(y_left) + len(y_right) * self._gini(y_right)) / n_samples
            
            if gini < best_gini:
                best_gini = gini
                best_split = (feat_idx, threshold)
        
        return best_split

    def _gini(self, y):
        """計算 Gini 不純度"""
        n_samples = len(y)
        if n_samples == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n_samples
        return 1.0 - np.sum(probabilities**2)

    def _calculate_leaf_value(self, y):
        """計算葉節點的類別機率分佈"""
        # 確保 y 不為空，如果為空則返回一個均勻分佈
        if len(y) == 0:
            return np.ones(self.n_classes_) / self.n_classes_

        # y 已經是 0 到 n_classes-1 的編碼
        counts = np.bincount(y, minlength=self.n_classes_)
        return counts / len(y)

    def predict(self, X):
            """
            預測 X 中樣本的類別標籤。

            參數:
            ----------
            X : array-like, shape (n_samples, n_features)
                要預測的樣本。

            返回:
            -------
            y_pred : array, shape (n_samples,)
                預測的類別標籤。
            """
            # 1. 檢查模型是否已經訓練過
            if not self.is_fitted_:
                raise RuntimeError("This ExtraTreeClassifier instance is not fitted yet. "
                                "Call 'fit' with appropriate arguments before using this estimator.")

            # 2. 確保 X 是 numpy 陣列
            X = np.asarray(X)

            # 3. 遍歷 X 中的每一個樣本，並使用 _traverse_tree 輔助函式進行預測
            predictions = [self._traverse_tree(sample, self.tree_) for sample in X]
            
            return np.array(predictions)

    def _traverse_tree(self, sample, node):
        """
        (輔助函式) 為單個樣本遞迴遍歷樹，並返回預測的類別。

        參數:
        ----------
        sample : array, shape (n_features,)
            單個輸入樣本。
        node : dict
            當前的樹節點 (來自 self.tree_)。

        返回:
        -------
        class_label : 
            預測的類別標籤。
        """
        # 基線條件 (Base case): 如果這是一個葉節點
        if node["leaf"]:
            # "value" 儲存的是各類別的機率分佈
            # 我們找到機率最高的那個類別的索引
            predicted_index = np.argmax(node["value"])
            # 使用 self.classes_ 將索引轉換回原始的類別標籤
            return self.classes_[predicted_index]

        # 遞迴條件 (Recursive case): 如果這是一個內部節點
        # 檢查樣本的特徵值並決定往左或往右
        if sample[node["feature"]] < node["threshold"]:
            return self._traverse_tree(sample, node["left"])
        else:
            return self._traverse_tree(sample, node["right"])
    def predict_proba(self, X):
        """Predict class probabilities (to be implemented by member C)."""
        raise NotImplementedError("predict_proba() method not yet implemented.")
   