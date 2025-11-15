"""
Comprehensive pytest test suite for Loss Function classes.

This test suite covers:
- MSE (Mean Squared Error)
- CrossEntropy_MultiClass

Tests include:
- Forward pass (loss computation)
- Derivative computation
- Parametrization for different inputs
- Edge cases and numerical stability
- Error handling for invalid inputs
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Module.Loss_Function import MSE, CrossEntropy_MultiClass


class TestMSE:
    """Test MSE (Mean Squared Error) loss class."""

    def test_mse_forward_basic(self):
        """Test MSE forward pass with basic inputs."""
        mse = MSE()
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_hat = np.array([[0.9, 0.1], [0.1, 0.9]])

        result = mse.forward(y, y_hat)

        # MSE = sum((y - y_hat)^2) / 2
        # = ((0.1)^2 + (0.1)^2 + (0.1)^2 + (0.1)^2) / 2
        # = (4 * 0.01) / 2 = 0.02
        expected = 0.02
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("y,y_hat,expected_loss", [
        (np.array([[1.0]]), np.array([[1.0]]), 0.0),  # Perfect prediction
        (np.array([[1.0]]), np.array([[0.0]]), 0.5),  # (1-0)^2 / 2 = 0.5
        (np.array([[0.0]]), np.array([[1.0]]), 0.5),  # (0-1)^2 / 2 = 0.5
        (np.array([[2.0]]), np.array([[0.0]]), 2.0),  # (2-0)^2 / 2 = 2.0
    ])
    def test_mse_forward_parametrized(self, y, y_hat, expected_loss):
        """Test MSE with parametrized inputs."""
        mse = MSE()
        result = mse.forward(y, y_hat)
        assert_allclose(result, expected_loss, rtol=1e-6)

    def test_mse_forward_perfect_prediction(self):
        """Test MSE returns 0 for perfect predictions."""
        mse = MSE()
        y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_hat = y.copy()

        result = mse.forward(y, y_hat)

        assert_allclose(result, 0.0, atol=1e-10)

    def test_mse_derivative_basic(self):
        """Test MSE derivative computation."""
        mse = MSE()
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_hat = np.array([[0.9, 0.1], [0.1, 0.9]])

        result = mse.derivative(y, y_hat)

        # Derivative: -(y - y_hat)
        expected = np.array([[-0.1, 0.1], [0.1, -0.1]])
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("y,y_hat", [
        (np.array([[1.0, 0.0]]), np.array([[0.5, 0.5]])),
        (np.array([[1.0, 2.0, 3.0]]), np.array([[1.5, 2.5, 3.5]])),
    ])
    def test_mse_derivative_parametrized(self, y, y_hat):
        """Test MSE derivative with parametrized inputs."""
        mse = MSE()
        result = mse.derivative(y, y_hat)
        expected = -(y - y_hat)
        assert_allclose(result, expected, rtol=1e-6)

    def test_mse_forward_shape_consistency(self):
        """Test that MSE returns scalar."""
        mse = MSE()
        y = np.random.randn(10, 5)
        y_hat = np.random.randn(10, 5)

        result = mse.forward(y, y_hat)

        assert np.isscalar(result) or result.shape == ()

    def test_mse_derivative_shape_consistency(self):
        """Test that MSE derivative maintains shape."""
        mse = MSE()
        y = np.random.randn(10, 5)
        y_hat = np.random.randn(10, 5)

        result = mse.derivative(y, y_hat)

        assert result.shape == y.shape

    def test_mse_no_side_effects(self):
        """Test that MSE doesn't modify input arrays."""
        mse = MSE()
        y = np.array([[1.0, 0.0]])
        y_hat = np.array([[0.9, 0.1]])
        y_original = y.copy()
        y_hat_original = y_hat.copy()

        mse.forward(y, y_hat)
        mse.derivative(y, y_hat)

        assert_allclose(y, y_original)
        assert_allclose(y_hat, y_hat_original)


class TestCrossEntropyMultiClass:
    """Test CrossEntropy_MultiClass loss class."""

    def test_cross_entropy_forward_basic(self):
        """Test cross-entropy forward pass with basic inputs."""
        ce = CrossEntropy_MultiClass()
        y = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_hat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

        result = ce.forward(y, y_hat)

        # Cross-entropy = -sum(y * log(y_hat))
        # = -(1*log(0.7) + 0 + 0 + 0 + 1*log(0.8) + 0)
        expected = -(np.log(0.7) + np.log(0.8))
        assert_allclose(result, expected, rtol=1e-6)

    def test_cross_entropy_forward_perfect_prediction(self):
        """Test cross-entropy with perfect predictions (should be near 0)."""
        ce = CrossEntropy_MultiClass()
        y = np.array([[1.0, 0.0, 0.0]])
        y_hat = np.array([[1.0, 0.0, 0.0]])

        result = ce.forward(y, y_hat)

        # With epsilon clipping, log(1 - epsilon) should be close to 0
        assert result < 1e-5  # Very small loss

    @pytest.mark.parametrize("epsilon", [1e-7, 1e-8, 1e-10])
    def test_cross_entropy_with_different_epsilon(self, epsilon):
        """Test cross-entropy with different epsilon values."""
        ce = CrossEntropy_MultiClass(epsilon=epsilon)
        y = np.array([[1.0, 0.0]])
        y_hat = np.array([[0.9, 0.1]])

        result = ce.forward(y, y_hat)

        # Should be finite and positive
        assert np.isfinite(result)
        assert result >= 0

    def test_cross_entropy_derivative_basic(self):
        """Test cross-entropy derivative computation."""
        ce = CrossEntropy_MultiClass()
        y = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_hat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

        result = ce.derivative(y, y_hat)

        # Derivative: y_hat - y
        expected = np.array([[-0.3, 0.2, 0.1], [0.1, -0.2, 0.1]])
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("y,y_hat", [
        (np.array([[1.0, 0.0]]), np.array([[0.8, 0.2]])),
        (np.array([[0.0, 1.0]]), np.array([[0.3, 0.7]])),
    ])
    def test_cross_entropy_derivative_parametrized(self, y, y_hat):
        """Test cross-entropy derivative with parametrized inputs."""
        ce = CrossEntropy_MultiClass()
        result = ce.derivative(y, y_hat)
        expected = y_hat - y
        assert_allclose(result, expected, rtol=1e-6)

    def test_cross_entropy_numerical_stability_zeros(self):
        """Test cross-entropy numerical stability when y_hat contains zeros."""
        ce = CrossEntropy_MultiClass(epsilon=1e-7)
        y = np.array([[1.0, 0.0, 0.0]])
        y_hat = np.array([[0.0, 0.5, 0.5]])  # Zero prediction for true class

        result = ce.forward(y, y_hat)

        # Should not be inf or nan due to epsilon clipping
        assert np.isfinite(result)

    def test_cross_entropy_numerical_stability_ones(self):
        """Test cross-entropy numerical stability when y_hat contains ones."""
        ce = CrossEntropy_MultiClass(epsilon=1e-7)
        y = np.array([[0.0, 1.0, 0.0]])
        y_hat = np.array([[0.0, 1.0, 0.0]])  # Perfect prediction

        result = ce.forward(y, y_hat)

        # Should be finite and very small
        assert np.isfinite(result)
        assert result < 1e-5

    def test_cross_entropy_forward_shape_consistency(self):
        """Test that cross-entropy returns scalar."""
        ce = CrossEntropy_MultiClass()
        y = np.random.rand(10, 5)
        y = y / y.sum(axis=1, keepdims=True)  # Normalize to probabilities
        y_hat = np.random.rand(10, 5)
        y_hat = y_hat / y_hat.sum(axis=1, keepdims=True)

        result = ce.forward(y, y_hat)

        assert np.isscalar(result) or result.shape == ()

    def test_cross_entropy_derivative_shape_consistency(self):
        """Test that cross-entropy derivative maintains shape."""
        ce = CrossEntropy_MultiClass()
        y = np.random.rand(10, 5)
        y_hat = np.random.rand(10, 5)

        result = ce.derivative(y, y_hat)

        assert result.shape == y.shape

    def test_cross_entropy_one_hot_encoding(self):
        """Test cross-entropy with one-hot encoded labels."""
        ce = CrossEntropy_MultiClass()

        # One-hot encoded labels
        y = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Predicted probabilities
        y_hat = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8]
        ])

        result = ce.forward(y, y_hat)

        # Cross-entropy should be positive and finite
        assert result > 0
        assert np.isfinite(result)

    def test_cross_entropy_no_side_effects(self):
        """Test that cross-entropy doesn't modify input arrays."""
        ce = CrossEntropy_MultiClass()
        y = np.array([[1.0, 0.0]])
        y_hat = np.array([[0.9, 0.1]])
        y_original = y.copy()
        y_hat_original = y_hat.copy()

        ce.forward(y, y_hat)
        ce.derivative(y, y_hat)

        assert_allclose(y, y_original)
        assert_allclose(y_hat, y_hat_original)


class TestLossFunctionIntegration:
    """Integration tests for loss functions."""

    @pytest.mark.parametrize("loss_class", [MSE, CrossEntropy_MultiClass])
    def test_loss_forward_returns_scalar(self, loss_class):
        """Test that all losses return scalar values."""
        loss = loss_class()
        y = np.random.rand(10, 5)
        y_hat = np.random.rand(10, 5)

        result = loss.forward(y, y_hat)

        assert np.isscalar(result) or result.shape == ()

    @pytest.mark.parametrize("loss_class", [MSE, CrossEntropy_MultiClass])
    def test_loss_derivative_preserves_shape(self, loss_class):
        """Test that all loss derivatives preserve shape."""
        loss = loss_class()
        y = np.random.rand(10, 5)
        y_hat = np.random.rand(10, 5)

        result = loss.derivative(y, y_hat)

        assert result.shape == y.shape

    def test_mse_vs_cross_entropy_behavior(self):
        """Test behavioral differences between MSE and cross-entropy."""
        y = np.array([[1.0, 0.0, 0.0]])
        y_hat = np.array([[0.7, 0.2, 0.1]])

        mse = MSE()
        ce = CrossEntropy_MultiClass()

        mse_loss = mse.forward(y, y_hat)
        ce_loss = ce.forward(y, y_hat)

        # Both should be positive for imperfect predictions
        assert mse_loss > 0
        assert ce_loss > 0

        # Both should return finite values
        assert np.isfinite(mse_loss)
        assert np.isfinite(ce_loss)
