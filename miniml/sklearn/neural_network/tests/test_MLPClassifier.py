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

from miniml.sklearn.neural_network.Module.Data_Generator import generate_classification_dataset


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


