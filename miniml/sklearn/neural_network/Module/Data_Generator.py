import numpy as np

from typing import Tuple


def generate_regression_dataset(num_samples: int = 3, X_dim: int = 10, y_dim: int = 3, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_seed)
    X = np.random.randn(num_samples, X_dim)

    weight_matrix = np.random.randn(X_dim, y_dim)
    bias_vector = np.random.randn(y_dim)

    logits = np.dot(X, weight_matrix) + bias_vector

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return X, y


def generate_classification_dataset(num_samples: int = 3, X_dim: int = 10, y_dim: int = 3, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_seed)
    weight_matrix = np.random.randn(X_dim, y_dim)
    bias_vector = np.random.randn(y_dim)

    X = np.random.randn(num_samples, X_dim)

    logits = np.dot(X, weight_matrix) + bias_vector

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    class_proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    class_idx = np.array([np.random.choice(y_dim, p=prob) for prob in class_proba])

    y = np.zeros((num_samples, y_dim))
    y[np.arange(num_samples), class_idx] = 1

    return X, y
