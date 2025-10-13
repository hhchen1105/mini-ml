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
        return 1 if v >= 0 else 0

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
        grads_W[-1] = np.outer(delta, activations[-2].T) / y.shape[0]
        grads_b[-1] = np.mean(delta, axis=0, keepdims=True)

        for i in reversed(range(len(self.weights) - 1)):
            delta = self._activation_derive(activations[i][0, 0]) * self.weights[i + 1].T @ delta
            grads_W[i] = np.outer(delta, activations[i].T) / y.shape[0]
            grads_b[i] = np.mean(delta, axis=0, keepdims=True)

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
                loss = np.mean((activations[-1] - y) ** 2)
                grads_W, grads_b = self._backward(activations, y[sample_index])

                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * grads_W[i]
                    self.biases[i] -= self.learning_rate * grads_b[i]

    def predict(self, X):
        X = np.asarray(X)
        activations = self._forward(X)
        return activations[-1][0, 0]
