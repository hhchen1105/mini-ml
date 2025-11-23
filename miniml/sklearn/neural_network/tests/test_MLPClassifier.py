"""
Pytest test suite for MLPClassifier class.

This test suite covers core functionality:
- fit method
- predict method
- predict_proba method
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniml.sklearn.neural_network import MLPClassifier


# ============== Data Generator ==============
def generate_classification_dataset(num_samples: int = 3, X_dim: int = 10, y_dim: int = 3, random_seed: int = 42) -> tuple:
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


class TestMLPClassifierFit:
    """Test MLPClassifier fit method."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_fit_with_different_batch_sizes(self, batch_size):
        """Test fit method with different batch sizes."""
        X, y = generate_classification_dataset(num_samples=100, X_dim=8, y_dim=3)

        # Store original shapes to check side effects
        original_X_shape = X.shape
        original_y_shape = y.shape

        clf = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            batch_size=batch_size,
            max_iter=5,
            verbose=False
        )

        clf.fit(X, y)

        # Check that network was initialized
        assert clf.is_initialized is True
        assert len(clf.layers) > 0

        # Check side effects - input data should not be modified
        assert X.shape == original_X_shape
        assert y.shape == original_y_shape

    @pytest.mark.parametrize("solver", ["adamw"])
    def test_fit_with_different_solvers(self, solver):
        """Test fit method with different solvers."""
        X, y = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3)

        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            solver=solver,
            max_iter=3,
            verbose=False
        )

        clf.fit(X, y)
        assert clf.is_initialized is True

    @pytest.mark.parametrize("loss", ["cross_entropy_multiclass", "mse"])
    def test_fit_with_different_loss_functions(self, loss):
        """Test fit method with different loss functions."""
        X, y = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3)

        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            loss=loss,
            max_iter=3,
            verbose=False
        )

        clf.fit(X, y)
        assert clf.is_initialized is True

    def test_fit_updates_weights(self):
        """Test that fit actually updates network weights."""
        X, y = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3)

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1, verbose=False)
        clf.initialize_network(X_dim=5, y_dim=3)

        # Store initial weights
        initial_weights = [layer.copy() for layer in clf.layers]

        # Fit the model
        clf.fit(X, y)

        # Check that weights have changed
        weights_changed = False
        for i, layer in enumerate(clf.layers):
            if not np.allclose(layer, initial_weights[i]):
                weights_changed = True
                break

        assert weights_changed, "Weights should be updated during training"


class TestMLPClassifierPredict:
    """Test MLPClassifier predict and predict_proba methods."""

    def test_predict_returns_class_labels(self):
        """Test that predict returns class labels."""
        X, y = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3)

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=5, verbose=False)
        clf.fit(X, y)

        predictions = clf.predict(X)

        # Check shape
        assert predictions.shape == (50,)

        # Check that predictions are valid class labels (0, 1, 2 for 3 classes)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 3)
        assert predictions.dtype in [np.int32, np.int64, int]

    def test_predict_proba_returns_probabilities(self):
        """Test that predict_proba returns valid probabilities."""
        X, y = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3)

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=5, verbose=False)
        clf.fit(X, y)

        probabilities = clf.predict_proba(X)

        # Check shape
        assert probabilities.shape == (50, 3)

        # Check that probabilities sum to 1
        prob_sums = np.sum(probabilities, axis=1)
        assert_allclose(prob_sums, np.ones(50), rtol=1e-5)

        # Check that all probabilities are in [0, 1]
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_predict_consistency_with_predict_proba(self):
        """Test that predict results are consistent with predict_proba."""
        X, y = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3)

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=5, verbose=False)
        clf.fit(X, y)

        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)

        # predict should return argmax of predict_proba
        expected_predictions = np.argmax(probabilities, axis=1)
        assert_array_equal(predictions, expected_predictions)

    @pytest.mark.parametrize("num_samples", [1, 10, 100])
    def test_predict_with_different_sample_sizes(self, num_samples):
        """Test predict with different numbers of samples."""
        X_train, y_train = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3, random_seed=42)
        X_test, _ = generate_classification_dataset(num_samples=num_samples, X_dim=5, y_dim=3, random_seed=123)

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=3, verbose=False)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        assert predictions.shape == (num_samples,)


class TestMLPClassifierErrorHandling:
    """Test error handling and edge cases for MLPClassifier."""

    # ============== fit() Error Handling Tests ==============
    @pytest.mark.parametrize("X_samples,y_samples", [
        (100, 50),   # X has more samples than y
        (50, 100),   # y has more samples than X
        (10, 5),     # Small dataset mismatch
    ])
    def test_fit_mismatched_sample_count(self, X_samples, y_samples):
        """Test that fit raises ValueError when X and y have different number of samples."""
        np.random.seed(42)
        X = np.random.randn(X_samples, 5)
        y = np.zeros((y_samples, 3))
        y[np.arange(y_samples), np.random.randint(0, 3, y_samples)] = 1

        clf = MLPClassifier(hidden_layer_sizes=(16,), verbose=False)

        with pytest.raises(ValueError, match="X and y must have the same number of samples"):
            clf.fit(X, y)

    def test_fit_y_not_one_hot_1d(self):
        """Test that fit raises ValueError when y is 1D array instead of 2D one-hot."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)  # 1D array of class labels

        clf = MLPClassifier(hidden_layer_sizes=(16,), verbose=False)

        with pytest.raises(ValueError, match="y must be one-hot encoded"):
            clf.fit(X, y)

    def test_fit_y_not_one_hot_3d(self):
        """Test that fit raises ValueError when y is 3D array."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 3, 1)  # 3D array

        clf = MLPClassifier(hidden_layer_sizes=(16,), verbose=False)

        with pytest.raises(ValueError, match="y must be one-hot encoded"):
            clf.fit(X, y)

    def test_fit_empty_dataset(self):
        """Test fit behavior with empty dataset."""
        X = np.empty((0, 5))
        y = np.empty((0, 3))

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1, verbose=False)

        # Empty dataset should work without error (edge case)
        clf.fit(X, y)
        assert clf.is_initialized is True

    # ============== predict() Error Handling Tests ==============
    def test_predict_before_fit(self):
        """Test that predict raises ValueError when called before fit."""
        X_test = np.random.randn(10, 5)

        clf = MLPClassifier(hidden_layer_sizes=(16,), verbose=False)

        with pytest.raises(ValueError, match="Network not initialized"):
            clf.predict(X_test)

    def test_predict_wrong_dimensions(self):
        """Test predict behavior when input has different feature dimensions than training data."""
        X_train, y_train = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3)
        X_test = np.random.randn(10, 10)  # 10 features instead of 5

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=3, verbose=False)
        clf.fit(X_train, y_train)

        # Should raise error due to dimension mismatch in numpy operations
        with pytest.raises((ValueError, IndexError)):
            clf.predict(X_test)

    # ============== predict_proba() Error Handling Tests ==============
    def test_predict_proba_before_fit(self):
        """Test that predict_proba raises ValueError when called before fit."""
        X_test = np.random.randn(10, 5)

        clf = MLPClassifier(hidden_layer_sizes=(16,), verbose=False)

        with pytest.raises(ValueError, match="Network not initialized"):
            clf.predict_proba(X_test)

    def test_predict_proba_wrong_dimensions(self):
        """Test predict_proba behavior when input has different feature dimensions than training data."""
        X_train, y_train = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3)
        X_test = np.random.randn(10, 8)  # 8 features instead of 5

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=3, verbose=False)
        clf.fit(X_train, y_train)

        # Should raise error due to dimension mismatch in numpy operations
        with pytest.raises((ValueError, IndexError)):
            clf.predict_proba(X_test)

    # ============== Edge Case Tests ==============
    def test_multiple_fit_calls_same_dimensions(self):
        """Test that calling fit multiple times with same dimensions works correctly."""
        X1, y1 = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3, random_seed=42)
        X2, y2 = generate_classification_dataset(num_samples=60, X_dim=5, y_dim=3, random_seed=123)

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2, verbose=False)

        # First fit
        clf.fit(X1, y1)
        first_weights = [layer.copy() for layer in clf.layers]

        # Second fit with same dimensions
        clf.fit(X2, y2)
        second_weights = [layer.copy() for layer in clf.layers]

        # Weights should have changed
        weights_changed = False
        for i in range(len(clf.layers)):
            if not np.allclose(first_weights[i], second_weights[i]):
                weights_changed = True
                break

        assert weights_changed, "Weights should be different after refitting"
        assert clf.is_initialized is True

    def test_multiple_fit_calls_different_dimensions(self):
        """Test that calling fit multiple times with different dimensions raises error.

        Current implementation does not reinitialize network when dimensions change,
        which causes a ValueError during forward pass due to shape mismatch.
        """
        X1, y1 = generate_classification_dataset(num_samples=50, X_dim=5, y_dim=3, random_seed=42)
        X2, y2 = generate_classification_dataset(num_samples=60, X_dim=8, y_dim=4, random_seed=123)

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2, verbose=False)

        # First fit
        clf.fit(X1, y1)
        assert len(clf.layers) == 2  # Input->Hidden, Hidden->Output
        assert clf.layers[0].shape[1] == 5 + 1  # 5 features + 1 bias
        assert clf.layers[-1].shape[0] == 3  # 3 output classes

        # Second fit with different dimensions should raise ValueError
        # because network is already initialized with different dimensions
        with pytest.raises(ValueError, match="shapes .* not aligned"):
            clf.fit(X2, y2)
