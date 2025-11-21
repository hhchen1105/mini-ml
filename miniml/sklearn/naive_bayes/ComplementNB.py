import numpy as np


class ComplementNB:
    def __init__(self, alpha=1.0, norm=False):
        self.alpha = alpha
        self.norm = norm

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if (X < 0).any():
            raise ValueError("ComplementNB only supports non-negative features")

        n_samples, n_features = X.shape

        # 1. 找出所有類別
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        self.n_features_in_ = n_features

        # 2. 算每個類別的樣本數
        self.class_count_ = np.bincount(
            y_indices, minlength=n_classes
        ).astype(float)

        # 3. 算每個類別的 feature 總數
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=float)
        for c in range(n_classes):
            self.feature_count_[c] = X[y_indices == c].sum(axis=0)

        # 4. 全部類別的 feature 總和
        self.feature_all_ = self.feature_count_.sum(axis=0)  # shape: (n_features,)

        # 5. complement 計數 + 平滑 (smoothing)
        alpha = float(self.alpha)
        comp_count = self.feature_all_ + alpha - self.feature_count_

        # 每個類別各自 normalize
        comp_norm = comp_count / comp_count.sum(axis=1, keepdims=True)

        # 取 log
        logged = np.log(comp_norm)

        # 6. 轉成權重
        if self.norm:
            summed = logged.sum(axis=1, keepdims=True)
            self.feature_log_prob_ = logged / summed
        else:
            # 跟 sklearn 做法一樣，用負號翻轉，方便後面做 argmax
            self.feature_log_prob_ = -logged

        # 7. 類別先驗機率（class prior）
        self.class_log_prior_ = np.log(
            self.class_count_ / self.class_count_.sum()
        )

        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but ComplementNB was fitted with "
                f"{self.n_features_in_} features"
            )

        # 分數 = X * 權重^T
        jll = X @ self.feature_log_prob_.T  # (n_samples, n_classes)


        if len(self.classes_) == 1:
            jll = jll + self.class_log_prior_

        return jll

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)

        # 用 log-sum-exp 做 softmax 得到機率
        max_jll = jll.max(axis=1, keepdims=True)
        exp_shifted = np.exp(jll - max_jll)
        proba = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = proba.argmax(axis=1)
        return self.classes_[indices]
