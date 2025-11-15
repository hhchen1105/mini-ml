import numpy as np

from typing import Tuple, Dict


class AdamWSolver:
    def __init__(self, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 1e-2):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m: Dict[Tuple, np.ndarray] = {}
        self.v: Dict[Tuple, np.ndarray] = {}
        self.t: Dict[Tuple, int] = {}

    def compute_update_value(self, layer_id: int, learning_rate: float, gradient: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if layer_id not in self.m:
            self.m[layer_id] = np.zeros_like(weights)
            self.v[layer_id] = np.zeros_like(weights)
            self.t[layer_id] = 0

        self.m[layer_id] = self.beta_1 * self.m[layer_id] + (1 - self.beta_1) * gradient
        self.v[layer_id] = self.beta_2 * self.v[layer_id] + (1 - self.beta_2) * (gradient ** 2)
        self.t[layer_id] += 1

        m_hat = self.m[layer_id] / (1 - self.beta_1 ** self.t[layer_id])
        v_hat = self.v[layer_id] / (1 - self.beta_2 ** self.t[layer_id])

        update_value = learning_rate * ((m_hat / (np.sqrt(v_hat) + self.epsilon)) + (self.weight_decay * weights))

        return update_value

    def initialize_solver(self) -> None:
        self.m.clear()
        self.v.clear()
        self.t.clear()
