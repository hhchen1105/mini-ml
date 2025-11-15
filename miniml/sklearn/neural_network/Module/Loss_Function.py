import numpy as np
from abc import ABC, abstractmethod


class BaseLoss(ABC):
    @abstractmethod
    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        pass

    @abstractmethod
    def derivative(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        pass


class MSE(BaseLoss):
    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum((y - y_hat) ** 2) / 2

    def derivative(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return -(y - y_hat)


class CrossEntropy_MultiClass(BaseLoss):
    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        y_hat_clipped = np.clip(y_hat, self.epsilon, 1 - self.epsilon)

        return np.sum(-np.log(y_hat_clipped) * y)

    def derivative(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return y_hat - y
