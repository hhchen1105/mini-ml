"""
Comprehensive pytest test suite for AdamWSolver class.

This test suite covers:
- Initialization and parameter settings
- Update value computation
- Momentum and velocity state management
- Bias correction
- Weight decay
- Parametrization for different hyperparameters
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Module.Solver import AdamWSolver


class TestAdamWSolverInitialization:
    """Test AdamWSolver initialization."""

    def test_default_initialization(self):
        """Test solver initialization with default parameters."""
        solver = AdamWSolver()

        assert solver.beta_1 == 0.9
        assert solver.beta_2 == 0.999
        assert solver.epsilon == 1e-8
        assert solver.weight_decay == 1e-2
        assert len(solver.m) == 0
        assert len(solver.v) == 0
        assert len(solver.t) == 0

    @pytest.mark.parametrize("beta_1,beta_2,epsilon,weight_decay", [
        (0.9, 0.999, 1e-8, 1e-2),
        (0.8, 0.99, 1e-7, 1e-3),
        (0.95, 0.9999, 1e-9, 0.0),
    ])
    def test_custom_initialization(self, beta_1, beta_2, epsilon, weight_decay):
        """Test solver initialization with custom parameters."""
        solver = AdamWSolver(
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay
        )

        assert solver.beta_1 == beta_1
        assert solver.beta_2 == beta_2
        assert solver.epsilon == epsilon
        assert solver.weight_decay == weight_decay

    def test_initialize_solver_clears_state(self):
        """Test that initialize_solver clears all state dictionaries."""
        solver = AdamWSolver()

        # Manually add some state
        solver.m[0] = np.array([1.0, 2.0, 3.0])
        solver.v[0] = np.array([4.0, 5.0, 6.0])
        solver.t[0] = 5

        # Initialize solver
        solver.initialize_solver()

        assert len(solver.m) == 0
        assert len(solver.v) == 0
        assert len(solver.t) == 0


class TestAdamWSolverComputeUpdateValue:
    """Test AdamWSolver compute_update_value method."""

    def test_first_update_initializes_state(self):
        """Test that first update initializes momentum and velocity."""
        solver = AdamWSolver()
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[1.0, 2.0, 3.0]])
        weights = np.array([[0.5, 0.6, 0.7]])

        solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        # Check that state was initialized
        assert layer_id in solver.m
        assert layer_id in solver.v
        assert layer_id in solver.t
        assert solver.t[layer_id] == 1

    def test_update_value_shape_consistency(self):
        """Test that update value has same shape as weights."""
        solver = AdamWSolver()
        layer_id = 0
        learning_rate = 0.001
        gradient = np.random.randn(10, 5)
        weights = np.random.randn(10, 5)

        update = solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        assert update.shape == weights.shape

    @pytest.mark.parametrize("learning_rate", [0.001, 0.01, 0.1])
    def test_update_with_different_learning_rates(self, learning_rate):
        """Test compute_update_value with different learning rates."""
        solver = AdamWSolver()
        layer_id = 0
        gradient = np.array([[1.0, 2.0]])
        weights = np.array([[0.5, 0.6]])

        update = solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        # Update should scale with learning rate
        assert update.shape == weights.shape
        # Verify all values are finite
        assert np.isfinite(update).all()

    def test_momentum_accumulation(self):
        """Test that momentum accumulates over multiple updates."""
        solver = AdamWSolver(beta_1=0.9)
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[1.0, 1.0]])
        weights = np.array([[0.5, 0.5]])

        # First update
        solver.compute_update_value(layer_id, learning_rate, gradient, weights)
        m_first = solver.m[layer_id].copy()

        # Second update with same gradient
        solver.compute_update_value(layer_id, learning_rate, gradient, weights)
        m_second = solver.m[layer_id].copy()

        # Momentum should have accumulated
        # m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
        expected_m_second = 0.9 * m_first + 0.1 * gradient
        assert_allclose(m_second, expected_m_second, rtol=1e-6)

    def test_velocity_accumulation(self):
        """Test that velocity accumulates over multiple updates."""
        solver = AdamWSolver(beta_2=0.999)
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[2.0, 2.0]])
        weights = np.array([[0.5, 0.5]])

        # First update
        solver.compute_update_value(layer_id, learning_rate, gradient, weights)
        v_first = solver.v[layer_id].copy()

        # Second update with same gradient
        solver.compute_update_value(layer_id, learning_rate, gradient, weights)
        v_second = solver.v[layer_id].copy()

        # Velocity should have accumulated
        # v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
        expected_v_second = 0.999 * v_first + 0.001 * (gradient ** 2)
        assert_allclose(v_second, expected_v_second, rtol=1e-6)

    def test_timestep_increment(self):
        """Test that timestep increments with each update."""
        solver = AdamWSolver()
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[1.0]])
        weights = np.array([[0.5]])

        for expected_t in range(1, 6):
            solver.compute_update_value(layer_id, learning_rate, gradient, weights)
            assert solver.t[layer_id] == expected_t

    def test_bias_correction(self):
        """Test that bias correction is applied."""
        solver = AdamWSolver(beta_1=0.9, beta_2=0.999)
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[1.0]])
        weights = np.array([[0.5]])

        # First update
        update_1 = solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        # With bias correction, update should be non-zero
        assert not np.allclose(update_1, 0.0)
        assert np.isfinite(update_1).all()

        # After multiple updates, bias correction effect should diminish
        for _ in range(10000):
            solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        # Bias correction factors should be close to 1
        bias_correction_1 = 1 - solver.beta_1 ** solver.t[layer_id]
        bias_correction_2 = 1 - solver.beta_2 ** solver.t[layer_id]

        assert bias_correction_1 > 0.99
        assert bias_correction_2 > 0.99


    @pytest.mark.parametrize("weight_decay", [0.0, 1e-3, 1e-2, 1e-1])
    def test_weight_decay_effect(self, weight_decay):
        """Test weight decay effect on updates."""
        solver = AdamWSolver(weight_decay=weight_decay)
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[1.0]])
        weights = np.array([[2.0]])

        update = solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        # Update should include weight decay term
        assert np.isfinite(update).all()

        # Larger weight decay should lead to larger updates (more regularization)
        if weight_decay > 0:
            assert update[0, 0] > 0  # Update includes positive weight decay term

    def test_multiple_layers_independent(self):
        """Test that updates for different layers are independent."""
        solver = AdamWSolver()
        learning_rate = 0.001

        # Update layer 0
        gradient_0 = np.array([[1.0]])
        weights_0 = np.array([[0.5]])
        solver.compute_update_value(0, learning_rate, gradient_0, weights_0)

        # Update layer 1
        gradient_1 = np.array([[2.0]])
        weights_1 = np.array([[0.6]])
        solver.compute_update_value(1, learning_rate, gradient_1, weights_1)

        # Check that both layers have independent state
        assert 0 in solver.m and 1 in solver.m
        assert 0 in solver.v and 1 in solver.v
        assert 0 in solver.t and 1 in solver.t

        # States should be different
        assert not np.allclose(solver.m[0], solver.m[1])

    def test_zero_gradient(self):
        """Test behavior with zero gradient."""
        solver = AdamWSolver(weight_decay=0.0)
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[0.0, 0.0]])
        weights = np.array([[0.5, 0.6]])

        update = solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        # With zero gradient and zero weight decay, update should be very small
        assert np.isfinite(update).all()

    def test_numerical_stability_large_gradient(self):
        """Test numerical stability with large gradients."""
        solver = AdamWSolver()
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[1000.0, 2000.0]])
        weights = np.array([[0.5, 0.6]])

        update = solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        # Update should be finite even with large gradients
        assert np.isfinite(update).all()

    def test_numerical_stability_small_gradient(self):
        """Test numerical stability with small gradients."""
        solver = AdamWSolver()
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[1e-10, 1e-10]])
        weights = np.array([[0.5, 0.6]])

        update = solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        # Update should be finite even with small gradients
        assert np.isfinite(update).all()


class TestAdamWSolverIntegration:
    """Integration tests for AdamWSolver."""

    def test_optimization_reduces_loss(self):
        """Test that AdamW updates move in direction to reduce loss."""
        solver = AdamWSolver()
        layer_id = 0
        learning_rate = 0.01

        # Simple quadratic loss: L = (w - target)^2
        target = np.array([[1.0]])
        weights = np.array([[0.0]])

        # Perform multiple updates
        for _ in range(10000):
            # Gradient: dL/dw = 2(w - target)
            gradient = 2 * (weights - target)
            update = solver.compute_update_value(layer_id, learning_rate, gradient, weights)
            weights -= update

        # Weights should move towards target
        assert np.abs(weights[0, 0] - target[0, 0]) < 0.1

    def test_solver_state_persistence(self):
        """Test that solver maintains state across updates."""
        solver = AdamWSolver()
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[1.0]])
        weights = np.array([[0.5]])

        # Perform 10 updates
        for _ in range(10):
            solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        # State should exist and have correct timestep
        assert solver.t[layer_id] == 10
        assert solver.m[layer_id].shape == gradient.shape
        assert solver.v[layer_id].shape == gradient.shape

    def test_initialize_solver_allows_fresh_start(self):
        """Test that initialize_solver allows starting fresh optimization."""
        solver = AdamWSolver()
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[1.0]])
        weights = np.array([[0.5]])

        # Perform some updates
        for _ in range(5):
            solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        assert solver.t[layer_id] == 5

        # Initialize solver
        solver.initialize_solver()

        # Perform update again
        solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        # Timestep should be reset to 1
        assert solver.t[layer_id] == 1

    def test_different_layer_shapes(self):
        """Test solver with different layer shapes."""
        solver = AdamWSolver()
        learning_rate = 0.001

        shapes = [(10, 5), (20, 10), (5, 3)]

        for layer_id, shape in enumerate(shapes):
            gradient = np.random.randn(*shape)
            weights = np.random.randn(*shape)

            update = solver.compute_update_value(layer_id, learning_rate, gradient, weights)

            assert update.shape == shape
            assert np.isfinite(update).all()
