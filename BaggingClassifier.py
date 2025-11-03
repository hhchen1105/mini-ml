import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

class MyBaggingClassifier:
    
    def __init__(self, 
                 base_estimator=None, 
                 n_estimators=10, 
                 max_samples=1.0, 
                 bootstrap=True, 
                 random_state=None):
        
        """
        BaggingClassifier 的建構子 (Initializer)

        參數 (Parameters):
        ----------
        base_estimator : object or None, default=None
            要進行 Bagging 的基底學習器。
            如果為 None，則預設使用 DecisionTreeClassifier。

        n_estimators : int, default=10
            要建立的基底學習器的數量。

        max_samples : int or float, default=1.0
            從 X 中抽樣以訓練每個基底學習器的樣本數。
            - 如果是 int, 則抽取 max_samples 個樣本。
            - 如果是 float, 則抽取 max_samples * X.shape[0] 個樣本。

        max_features : int or float, default=1.0
            從 X 中抽樣以訓練每個基底學習器的特徵數。
            - 如果是 int, 則抽取 max_features 個特徵。
            - 如果是 float, 則抽取 max_features * X.shape[1] 個特徵。

        bootstrap : bool, default=True
            是否對樣本進行「有放回抽樣」(Bootstrap)。
            如果為 False，則進行「無放回抽樣」(Pasting)。

        random_state : int, default=None
            用於控制隨機性的種子，確保結果可以重現。
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        if self.base_estimator is None:
            self._estimator = DecisionTreeClassifier()
        else:
            self._estimator = self.base_estimator
        
        self.estimators_ = []
        self._rng = np.random.RandomState(self.random_state)

    def fit(self, X, y):

        self.estimators_ = []
        n_samples = X.shape[0]

        if isinstance(self.max_samples, float):
            n_draw_samples = int(self.max_samples * n_samples)
        elif isinstance(self.max_samples, int):
            n_draw_samples = self.max_samples
        

        for _ in range(self.n_estimators):
            indices = self._rng.choice(n_samples, 
                                       size=n_draw_samples, 
                                       replace=self.bootstrap)
            
            X_sample = X[indices]
            y_sample = y[indices]
            
            model = clone(self._estimator)
            
            model.fit(X_sample, y_sample)
            
            self.estimators_.append(model)
            
        return self
        