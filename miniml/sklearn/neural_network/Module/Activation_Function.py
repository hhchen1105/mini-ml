import numpy as np
from abc import ABC, abstractmethod


class BaseActivation(ABC):
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, X: np.ndarray) -> np.ndarray:
        pass


class SigmoidActivation(BaseActivation):
    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(X, -88, 88)
        return np.where(X >= 0, 1 / (1 + np.exp(-X)), np.exp(X) / (1 + np.exp(X)))

    def derivative(self, X: np.ndarray) -> np.ndarray:
        X = self.forward(X)
        return X * (1 - X)


class LeakyReLUActivation(BaseActivation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.where(X >= 0, X, self.alpha * X)

    def derivative(self, X: np.ndarray) -> np.ndarray:
        return np.where(X >= 0, 1.0, self.alpha)


class IdentityActivation(BaseActivation):
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X

    def derivative(self, X: np.ndarray) -> np.ndarray:
        return np.ones_like(X)


class SoftmaxActivation(BaseActivation):
    def forward(self, X: np.ndarray) -> np.ndarray:
        X_max = np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X - X_max)

        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def derivative(self, X: np.ndarray) -> np.ndarray:
        X = self.forward(X)
        current_batch_size, num_classes = X.shape

        jacobian = np.zeros((current_batch_size, num_classes, num_classes))

        for i in range(current_batch_size):
            s = X[i].reshape(-1, 1)  # 列向量
            jacobian[i] = np.diag(s.flatten()) - np.dot(s, s.T)

        return jacobian
