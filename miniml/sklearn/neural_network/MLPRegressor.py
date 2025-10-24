import numpy as np

class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter

    def _initialize_weights(self, n_features, n_outputs):
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(2 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * limit
            b = np.zeros((layer_sizes[i + 1], 1))
            self.weights.append(W)
            self.biases.append(b)

    def _activation(self, v):
        return np.maximum(0, v)

    def _activation_derive(self, v):
        return (v > 0).astype(float) # 改成對"整個陣列"計算

    def _forward(self, X):
        activations = [X.reshape(-1, 1)]
        for i in range(len(self.weights) - 1):
            v = self.weights[i] @ activations[-1] + self.biases[i]
            y = self._activation(v)
            activations.append(y)
        v = self.weights[-1] @ activations[-1] + self.biases[-1]
        activations.append(v)
        return activations

    def _backward(self, activations, y):
       	grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        delta = activations[-1] - y
        grads_W[-1] = delta @ activations[-2].T # 1. 要做的是delta跟輸出值 2. 每次只有一個樣本，不需要除以樣本數
        grads_b[-1] = delta # 每次只有一個樣本 delta 已經是 (神經元數, 1) 形狀

        for i in reversed(range(len(self.weights) - 1)):
            delta = self._activation_derive(activations[i + 1]) * (self.weights[i + 1].T @ delta)
            grads_W[i] = delta @ activations[i].T
            grads_b[i] = delta

        return grads_W, grads_b

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        n_samples, n_features = X.shape
        n_outputs = y.shape[1] #?
        self._initialize_weights(n_features, n_outputs)

        for epoch in range(self.max_iter):
            for sample_index in range(len(X)):
                activations = self._forward(X[sample_index])
                loss = np.mean((activations[-1] - y[sample_index]) ** 2) # 是要跟目前處理的這筆y比較
                grads_W, grads_b = self._backward(activations, y[sample_index])

                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * grads_W[i]
                    self.biases[i] -= self.learning_rate * grads_b[i]
        
        # 儲存 w 跟 b，符合 sklearn 的命名，讓它通過檢查
        self.coefs_ = self.weights
        self.intercepts_ = self.biases

    def predict(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            activations = self._forward(X)
            return activations[-1][0, 0] # 當輸出層不只一個節點時，可能需要調整成用flatten?
        else: # 官方版本支援批次預測
            y_preds = []
            for sample_index in range(len(X)):
                activations = self._forward(X[sample_index])
                y_preds.append(activations[-1][0, 0]) # 同上
            return np.array(y_preds)
