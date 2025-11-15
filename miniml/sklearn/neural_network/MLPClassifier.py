import numpy as np

from math import sqrt
from typing import Tuple, List

from miniml.sklearn.neural_network.Module.Utils import Activations, LearningRates, Solvers, Losses


BIAS_DIM = 1
BIAS_TERM = 1
HE_FACTOR = 2.0


class MLPClassifier:
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
