"""
Comprehensive pytest test suite for Activation Function classes.

This test suite covers:
- SigmoidActivation
- LeakyReLUActivation
- IdentityActivation
- SoftmaxActivation

Tests include:
- Forward pass correctness
- Derivative correctness
- Parametrization for different inputs
- Edge cases and numerical stability
- Shape consistency
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Module.Activation_Function import (
    SigmoidActivation,
    LeakyReLUActivation,
    IdentityActivation,
    SoftmaxActivation
)


class TestSigmoidActivation:
    """Test SigmoidActivation class."""

    def test_sigmoid_forward_basic(self):
        """Test sigmoid forward pass with basic inputs."""
        sigmoid = SigmoidActivation()
        X = np.array([[0.0, 1.0, -1.0]])

        result = sigmoid.forward(X)

        # Sigmoid(0) = 0.5, Sigmoid(1) ≈ 0.731, Sigmoid(-1) ≈ 0.269
        expected = np.array([[0.5, 0.7310585786, 0.2689414214]])
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("X,expected", [
        (np.array([[0.0]]), np.array([[0.5]])),
        (np.array([[100.0]]), np.array([[1.0]])),  # Large positive
        (np.array([[-100.0]]), np.array([[0.0]])),  # Large negative
    ])
    def test_sigmoid_forward_parametrized(self, X, expected):
        """Test sigmoid with parametrized inputs."""
        sigmoid = SigmoidActivation()
        result = sigmoid.forward(X)
        assert_allclose(result, expected, atol=1e-6)

    def test_sigmoid_derivative_basic(self):
        """Test sigmoid derivative."""
        sigmoid = SigmoidActivation()
        X = np.array([[0.0, 1.0, -1.0]])

        result = sigmoid.derivative(X)

        # Derivative at 0: 0.25, at 1: ~0.196, at -1: ~0.196
        expected = np.array([[0.25, 0.19661193, 0.19661193]])
        assert_allclose(result, expected, rtol=1e-6)

    def test_sigmoid_numerical_stability_large_positive(self):
        """Test sigmoid numerical stability with large positive values."""
        sigmoid = SigmoidActivation()
        X = np.array([[1000.0]])

        result = sigmoid.forward(X)

        # Should not overflow, should be close to 1
        assert np.isfinite(result).all()
        assert_allclose(result, [[1.0]], atol=1e-6)

    def test_sigmoid_numerical_stability_large_negative(self):
        """Test sigmoid numerical stability with large negative values."""
        sigmoid = SigmoidActivation()
        X = np.array([[-1000.0]])

        result = sigmoid.forward(X)

        # Should not underflow, should be close to 0
        assert np.isfinite(result).all()
        assert_allclose(result, [[0.0]], atol=1e-6)

    def test_sigmoid_shape_consistency(self):
        """Test that sigmoid maintains input shape."""
        sigmoid = SigmoidActivation()
        X = np.random.randn(10, 5)

        result = sigmoid.forward(X)

        assert result.shape == X.shape

    def test_sigmoid_output_range(self):
        """Test that sigmoid output is in (0, 1)."""
        sigmoid = SigmoidActivation()
        X = np.random.randn(100, 10) * 10  # Large random values

        result = sigmoid.forward(X)

        assert np.all(result >= 0)
        assert np.all(result <= 1)


class TestLeakyReLUActivation:
    """Test LeakyReLUActivation class."""

    def test_leaky_relu_forward_default_alpha(self):
        """Test LeakyReLU forward pass with default alpha."""
        leaky_relu = LeakyReLUActivation()
        X = np.array([[1.0, 0.0, -1.0, 2.0, -2.0]])

        result = leaky_relu.forward(X)

        # For x >= 0: f(x) = x, for x < 0: f(x) = 0.01 * x
        expected = np.array([[1.0, 0.0, -0.01, 2.0, -0.02]])
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("alpha", [0.01, 0.1, 0.2, 0.3])
    def test_leaky_relu_forward_custom_alpha(self, alpha):
        """Test LeakyReLU with different alpha values."""
        leaky_relu = LeakyReLUActivation(alpha=alpha)
        X = np.array([[-1.0, 1.0]])

        result = leaky_relu.forward(X)

        expected = np.array([[-alpha, 1.0]])
        assert_allclose(result, expected, rtol=1e-6)

    def test_leaky_relu_derivative_default_alpha(self):
        """Test LeakyReLU derivative with default alpha."""
        leaky_relu = LeakyReLUActivation()
        X = np.array([[1.0, 0.0, -1.0, 2.0, -2.0]])

        result = leaky_relu.derivative(X)

        # For x >= 0: derivative = 1, for x < 0: derivative = alpha
        expected = np.array([[1.0, 1.0, 0.01, 1.0, 0.01]])
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("alpha,X,expected_deriv", [
        (0.01, np.array([[5.0, -5.0]]), np.array([[1.0, 0.01]])),
        (0.1, np.array([[3.0, -3.0]]), np.array([[1.0, 0.1]])),
        (0.2, np.array([[0.0]]), np.array([[1.0]])),
    ])
    def test_leaky_relu_derivative_parametrized(self, alpha, X, expected_deriv):
        """Test LeakyReLU derivative with parametrized inputs."""
        leaky_relu = LeakyReLUActivation(alpha=alpha)
        result = leaky_relu.derivative(X)
        assert_allclose(result, expected_deriv, rtol=1e-6)

    def test_leaky_relu_shape_consistency(self):
        """Test that LeakyReLU maintains input shape."""
        leaky_relu = LeakyReLUActivation()
        X = np.random.randn(20, 15)

        result = leaky_relu.forward(X)

        assert result.shape == X.shape

    def test_leaky_relu_no_side_effects(self):
        """Test that LeakyReLU doesn't modify input array."""
        leaky_relu = LeakyReLUActivation()
        X = np.array([[1.0, -1.0, 0.0]])
        X_original = X.copy()

        leaky_relu.forward(X)

        assert_allclose(X, X_original)


class TestIdentityActivation:
    """Test IdentityActivation class."""

    def test_identity_forward(self):
        """Test identity forward pass returns input unchanged."""
        identity = IdentityActivation()
        X = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])

        result = identity.forward(X)

        assert_allclose(result, X)

    @pytest.mark.parametrize("shape", [
        (1, 1),
        (10, 5),
        (100, 50),
        (5, 10, 3)  # 3D array
    ])
    def test_identity_forward_different_shapes(self, shape):
        """Test identity with different input shapes."""
        identity = IdentityActivation()
        X = np.random.randn(*shape)

        result = identity.forward(X)

        assert_allclose(result, X)

    def test_identity_derivative(self):
        """Test identity derivative returns ones."""
        identity = IdentityActivation()
        X = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])

        result = identity.derivative(X)

        expected = np.ones_like(X)
        assert_allclose(result, expected)

    def test_identity_derivative_shape(self):
        """Test that derivative has same shape as input."""
        identity = IdentityActivation()
        X = np.random.randn(15, 10)

        result = identity.derivative(X)

        assert result.shape == X.shape
        assert_allclose(result, np.ones_like(X))


class TestSoftmaxActivation:
    """Test SoftmaxActivation class."""

    def test_softmax_forward_basic(self):
        """Test softmax forward pass with basic inputs."""
        softmax = SoftmaxActivation()
        X = np.array([[1.0, 2.0, 3.0]])

        result = softmax.forward(X)

        # Softmax should sum to 1
        assert_allclose(np.sum(result, axis=1), [1.0], rtol=1e-6)

        # Check approximate values
        expected = np.array([[0.09003057, 0.24472847, 0.66524096]])
        assert_allclose(result, expected, rtol=1e-6)

    def test_softmax_forward_probabilities_sum_to_one(self):
        """Test that softmax output sums to 1 for each sample."""
        softmax = SoftmaxActivation()
        X = np.random.randn(10, 5)

        result = softmax.forward(X)

        sums = np.sum(result, axis=1)
        assert_allclose(sums, np.ones(10), rtol=1e-6)

    @pytest.mark.parametrize("num_samples,num_classes", [
        (1, 3),
        (10, 5),
        (100, 10),
    ])
    def test_softmax_forward_different_sizes(self, num_samples, num_classes):
        """Test softmax with different input sizes."""
        softmax = SoftmaxActivation()
        X = np.random.randn(num_samples, num_classes)

        result = softmax.forward(X)

        assert result.shape == (num_samples, num_classes)
        assert_allclose(np.sum(result, axis=1), np.ones(num_samples), rtol=1e-6)

    def test_softmax_forward_numerical_stability(self):
        """Test softmax numerical stability with large values."""
        softmax = SoftmaxActivation()
        X = np.array([[1000.0, 1001.0, 999.0]])

        result = softmax.forward(X)

        # Should not overflow
        assert np.isfinite(result).all()
        assert_allclose(np.sum(result, axis=1), [1.0], rtol=1e-6)

    def test_softmax_forward_output_range(self):
        """Test that softmax output is in [0, 1]."""
        softmax = SoftmaxActivation()
        X = np.random.randn(50, 10) * 100  # Large random values

        result = softmax.forward(X)

        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_softmax_derivative_shape(self):
        """Test softmax derivative returns correct Jacobian shape."""
        softmax = SoftmaxActivation()
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = softmax.derivative(X)

        # Jacobian shape: (batch_size, num_classes, num_classes)
        assert result.shape == (2, 3, 3)

    def test_softmax_derivative_properties(self):
        """Test mathematical properties of softmax derivative."""
        softmax = SoftmaxActivation()
        X = np.array([[1.0, 2.0, 3.0]])

        jacobian = softmax.derivative(X)

        # Jacobian should be square for each sample
        assert jacobian.shape == (1, 3, 3)

        # For softmax, sum of each row in Jacobian should be 0
        row_sums = np.sum(jacobian[0], axis=1)
        assert_allclose(row_sums, np.zeros(3), atol=1e-10)

    def test_softmax_forward_uniform_input(self):
        """Test softmax with uniform input values."""
        softmax = SoftmaxActivation()
        X = np.array([[1.0, 1.0, 1.0]])

        result = softmax.forward(X)

        # With uniform input, output should be uniform
        expected = np.array([[1/3, 1/3, 1/3]])
        assert_allclose(result, expected, rtol=1e-6)

    def test_softmax_no_side_effects(self):
        """Test that softmax doesn't modify input array."""
        softmax = SoftmaxActivation()
        X = np.array([[1.0, 2.0, 3.0]])
        X_original = X.copy()

        softmax.forward(X)

        assert_allclose(X, X_original)


class TestActivationFunctionIntegration:
    """Integration tests for activation functions."""

    @pytest.mark.parametrize("activation_class", [
        SigmoidActivation,
        LeakyReLUActivation,
        IdentityActivation,
        SoftmaxActivation
    ])
    def test_all_activations_preserve_batch_size(self, activation_class):
        """Test that all activations preserve batch size dimension."""
        activation = activation_class()
        X = np.random.randn(32, 10)

        result = activation.forward(X)

        assert result.shape[0] == 32  # Batch size preserved

    @pytest.mark.parametrize("activation_class", [
        SigmoidActivation,
        LeakyReLUActivation,
        IdentityActivation,
    ])
    def test_element_wise_activations_preserve_shape(self, activation_class):
        """Test that element-wise activations preserve full shape."""
        activation = activation_class()
        X = np.random.randn(15, 20)

        result = activation.forward(X)

        assert result.shape == X.shape
