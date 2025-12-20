import numpy as np

from math import sqrt, pi, cos
from typing import Tuple, List
from abc import ABC, abstractmethod
from miniml.sklearn.base import BaseEstimator


# ============== Activation Functions ==============
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
            s = X[i].reshape(-1, 1)
            jacobian[i] = np.diag(s.flatten()) - np.dot(s, s.T)
        return jacobian


# ============== Loss Functions ==============
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


# ============== Learning Rate Scheduler ==============
class WarmUpCosineAnnealing:
    def __init__(self, lr_min: float, lr_max: float, warm_up: int = 10):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up

    def compute_lr(self, current_iter: int, max_iter: int, current_lr: float) -> float:
        if self.lr_min == self.lr_max:
            return self.lr_max
        
        if current_iter < self.warm_up:
            current_lr = self.lr_max / self.warm_up * (current_iter + 1)
        else:
            max_iter -= self.warm_up
            current_iter -= self.warm_up
            current_lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1 + cos(current_iter * pi / max_iter))

        return current_lr


# ============== Solver ==============
class AdamWSolver:
    def __init__(self, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 1e-2):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = {}

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


# ============== Activation/Loss/Solver Registry ==============
class Activations:
    ACTIVATION_FUNCTION_MAP = {
        "sigmoid": SigmoidActivation,
        "leaky_relu": LeakyReLUActivation,
        "identity": IdentityActivation,
        "softmax": SoftmaxActivation
    }

    @classmethod
    def get_activation(cls, activation: str, **kwargs) -> object:
        return cls.ACTIVATION_FUNCTION_MAP[activation](**kwargs)


class LearningRates:
    LEARNING_RATE_MAP = {"warmup_cosine_annealing": WarmUpCosineAnnealing}

    @classmethod
    def get_learning_rate(cls, learning_rate: str, **kwargs) -> object:
        return cls.LEARNING_RATE_MAP[learning_rate](**kwargs)


class Solvers:
    SOLVER_MAP = {"adamw": AdamWSolver}

    @classmethod
    def get_solver(cls, solver: str, **kwargs) -> object:
        return cls.SOLVER_MAP[solver](**kwargs)


class Losses:
    LOSS_MAP = {
        "mse": MSE,
        "cross_entropy_multiclass": CrossEntropy_MultiClass
    }

    @classmethod
    def get_loss(cls, loss: str, **kwargs) -> object:
        return cls.LOSS_MAP[loss](**kwargs)


# ============== Constants ==============
BIAS_DIM = 1
BIAS_TERM = 1
HE_FACTOR = 2.0


# ============== MLPClassifier ==============
class MLPClassifier(BaseEstimator):
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (32,), solver: str = "adamw", batch_size: int = 1,
                 learning_rate: str = "warmup_cosine_annealing", learning_rate_init: Tuple[float, float] = (1e-3, 0.1),
                 loss: str = "cross_entropy_multiclass", max_iter: int = 100, verbose: bool = True):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.is_initialized = False

        self.layers = []
        self.leaky_relu = Activations.get_activation("leaky_relu")
        self.softmax = Activations.get_activation("softmax")

        lr_min = learning_rate_init[0]
        lr_max = learning_rate_init[1]
        self.learning_rate = LearningRates.get_learning_rate(learning_rate, lr_min=lr_min, lr_max=lr_max)

        self.solver = Solvers.get_solver(solver)
        self.loss = Losses.get_loss(loss)

    def initialize_network(self, X_dim: int, y_dim: int) -> None:
        # Build architecture: Input Layer  -> Hidden layers  -> Output Layer
        layer_sizes = (X_dim,) + self.hidden_layer_sizes + (y_dim,)

        self.layers = []
        for layer_idx in range(1, len(layer_sizes)):
            num_weights = layer_sizes[layer_idx-1]
            num_neurons = layer_sizes[layer_idx]
            layer = np.random.randn(num_neurons, num_weights + BIAS_DIM) * sqrt(HE_FACTOR / num_weights)
            self.layers.append(layer)

        self.is_initialized = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict class
        y_hats = self.predict_proba(X)
        return np.argmax(y_hats, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Predict probability of each class
        y_hats, _ = self.forward(X)
        return y_hats[-1]

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not self.is_initialized:
            raise ValueError("Network not initialized.")

        # Forward pass
        X = np.append(X, np.full((X.shape[0], 1), BIAS_TERM), axis=1)

        # Caches used for backward
        logits = []
        y_hats = [X]

        # Pass through hidden layers
        y_hat = X
        for layer_idx in range(len(self.layers) - 1):
            logit = np.dot(y_hat, self.layers[layer_idx].T)
            y_hat = self.leaky_relu.forward(logit)
            y_hat = np.append(y_hat, np.full((y_hat.shape[0], 1), BIAS_TERM), axis=1)

            logits.append(logit)
            y_hats.append(y_hat)

        # Pass through output layer
        logit = np.dot(y_hat, self.layers[-1].T)
        y_hat = self.softmax.forward(logit)

        logits.append(logit)
        y_hats.append(y_hat)

        return y_hats, logits

    def backward(self, y_hats: List[np.ndarray], logits: List[np.ndarray], y: np.ndarray, current_batch_size: int) -> List[np.ndarray]:
        # Backward pass
        gradients = [np.zeros_like(layer) for layer in self.layers]

        # Process neurons in output layer
        deltas = y_hats[-1] - y
        gradients[-1] = np.dot(deltas.T, y_hats[-2]) / current_batch_size

        for layer_idx in range(len(self.layers) - 2, -1, -1):
            # Process neurons in hidden layers
            deltas = np.dot(deltas, self.layers[layer_idx + 1][:, :-1]) * self.leaky_relu.derivative(logits[layer_idx])
            gradients[layer_idx] = np.dot(deltas.T, y_hats[layer_idx]) / current_batch_size

        return gradients

    def update_weights(self, gradients: List[np.ndarray], current_lr: float) -> None:
        # Update weights of all neurons
        for layer_idx in range(len(self.layers)):
            gradient = gradients[layer_idx]
            weight = self.layers[layer_idx]
            update_value = self.solver.compute_update_value(layer_idx, current_lr, gradient, weight)
            self.layers[layer_idx] -= update_value

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if len(y.shape) != 2:
            raise ValueError("y must be one-hot encoded")

        # Train the model
        num_samples, X_dim = X.shape
        y_dim = y.shape[1]
        current_lr = 0.0

        self.solver.initialize_solver()

        if not self.is_initialized:
            self.initialize_network(X_dim, y_dim)

        # Training loop
        for current_iter in range(self.max_iter):
            total_loss = 0.0
            total_samples_processed = 0

            # Compute learning rate
            current_lr = self.learning_rate.compute_lr(current_iter, self.max_iter, current_lr)

            # Process each batch
            for sample_idx in range(0, num_samples, self.batch_size):
                # Get current batch (no padding)
                current_X = X[sample_idx:sample_idx + self.batch_size]
                current_y = y[sample_idx:sample_idx + self.batch_size]
                current_batch_size = current_X.shape[0]

                # Forward pass
                y_hats, logits = self.forward(current_X)

                # Compute loss
                batch_loss = self.loss.forward(current_y, y_hats[-1])
                total_loss += batch_loss
                total_samples_processed += current_batch_size

                # Backward pass
                gradients = self.backward(y_hats, logits, current_y, current_batch_size)
                self.update_weights(gradients, current_lr)

            if self.verbose:
                # Calculate average loss
                avg_loss = total_loss / total_samples_processed if total_samples_processed > 0 else 0
                print(f"Epoch {current_iter + 1:>3} / {self.max_iter}, Average Loss: {avg_loss:.6f}")