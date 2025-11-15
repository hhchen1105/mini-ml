"""
Comprehensive pytest test suite for Utils factory classes.

This test suite covers:
- Activations factory
- LearningRates factory
- Solvers factory
- Losses factory

Tests include:
- Factory method correctness
- Parametrization for all supported types
- Error handling for invalid types
- Instance verification
- Custom parameters passing
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Module.Utils import Activations, LearningRates, Solvers, Losses
from Module.Activation_Function import (
    SigmoidActivation,
    LeakyReLUActivation,
    IdentityActivation,
    SoftmaxActivation
)
from Module.Loss_Function import MSE, CrossEntropy_MultiClass
from Module.Solver import AdamWSolver
from Module.Learning_Rate_Scheduler import WarmUpCosineAnnealing


class TestActivationsFactory:
    """Test Activations factory class."""

    @pytest.mark.parametrize("activation_name,expected_class", [
        ("sigmoid", SigmoidActivation),
        ("leaky_relu", LeakyReLUActivation),
        ("identity", IdentityActivation),
        ("softmax", SoftmaxActivation),
    ])
    def test_get_activation_returns_correct_type(self, activation_name, expected_class):
        """Test that get_activation returns correct activation class."""
        activation = Activations.get_activation(activation_name)

        assert isinstance(activation, expected_class)

    def test_sigmoid_activation_instance(self):
        """Test getting sigmoid activation instance."""
        activation = Activations.get_activation("sigmoid")

        assert isinstance(activation, SigmoidActivation)

        # Test that it works
        X = np.array([[0.0]])
        result = activation.forward(X)
        assert np.isclose(result[0, 0], 0.5)

    def test_leaky_relu_activation_instance(self):
        """Test getting leaky ReLU activation instance."""
        activation = Activations.get_activation("leaky_relu")

        assert isinstance(activation, LeakyReLUActivation)

        # Test default alpha
        assert activation.alpha == 0.01

    @pytest.mark.parametrize("alpha", [0.01, 0.1, 0.2])
    def test_leaky_relu_with_custom_alpha(self, alpha):
        """Test getting leaky ReLU with custom alpha parameter."""
        activation = Activations.get_activation("leaky_relu", alpha=alpha)

        assert isinstance(activation, LeakyReLUActivation)
        assert activation.alpha == alpha

    def test_identity_activation_instance(self):
        """Test getting identity activation instance."""
        activation = Activations.get_activation("identity")

        assert isinstance(activation, IdentityActivation)

        # Test that it works (identity function)
        X = np.array([[1.0, 2.0, 3.0]])
        result = activation.forward(X)
        assert np.allclose(result, X)

    def test_softmax_activation_instance(self):
        """Test getting softmax activation instance."""
        activation = Activations.get_activation("softmax")

        assert isinstance(activation, SoftmaxActivation)

        # Test that it produces valid probabilities
        X = np.array([[1.0, 2.0, 3.0]])
        result = activation.forward(X)
        assert np.isclose(np.sum(result), 1.0)

    def test_invalid_activation_raises_keyerror(self):
        """Test that invalid activation name raises KeyError."""
        with pytest.raises(KeyError):
            Activations.get_activation("invalid_activation")

    def test_activation_map_contains_all_activations(self):
        """Test that ACTIVATION_FUNCTION_MAP contains expected activations."""
        expected_keys = {"sigmoid", "leaky_relu", "identity", "softmax"}
        actual_keys = set(Activations.ACTIVATION_FUNCTION_MAP.keys())

        assert expected_keys == actual_keys


class TestLearningRatesFactory:
    """Test LearningRates factory class."""

    def test_get_warmup_cosine_annealing(self):
        """Test getting WarmUpCosineAnnealing scheduler."""
        scheduler = LearningRates.get_learning_rate(
            "warmup_cosine_annealing",
            lr_min=1e-3,
            lr_max=0.1
        )

        assert isinstance(scheduler, WarmUpCosineAnnealing)
        assert scheduler.lr_min == 1e-3
        assert scheduler.lr_max == 0.1

    @pytest.mark.parametrize("lr_min,lr_max,warm_up", [
        (1e-4, 0.1, 5),
        (1e-3, 0.05, 10),
        (1e-5, 0.5, 20),
    ])
    def test_warmup_cosine_annealing_with_parameters(self, lr_min, lr_max, warm_up):
        """Test WarmUpCosineAnnealing with custom parameters."""
        scheduler = LearningRates.get_learning_rate(
            "warmup_cosine_annealing",
            lr_min=lr_min,
            lr_max=lr_max,
            warm_up=warm_up
        )

        assert isinstance(scheduler, WarmUpCosineAnnealing)
        assert scheduler.lr_min == lr_min
        assert scheduler.lr_max == lr_max
        assert scheduler.warm_up == warm_up

    def test_invalid_learning_rate_raises_keyerror(self):
        """Test that invalid learning rate name raises KeyError."""
        with pytest.raises(KeyError):
            LearningRates.get_learning_rate("invalid_scheduler")

    def test_learning_rate_map_contains_expected_schedulers(self):
        """Test that LEARNING_RATE_MAP contains expected schedulers."""
        expected_keys = {"warmup_cosine_annealing"}
        actual_keys = set(LearningRates.LEARNING_RATE_MAP.keys())

        assert expected_keys == actual_keys

    def test_scheduler_functionality(self):
        """Test that returned scheduler actually works."""
        scheduler = LearningRates.get_learning_rate(
            "warmup_cosine_annealing",
            lr_min=1e-3,
            lr_max=0.1,
            warm_up=10
        )

        # Test compute_lr method
        lr = scheduler.compute_lr(current_iter=0, max_iter=100, current_lr=0.0)

        assert lr > 0
        assert lr <= 0.1


class TestSolversFactory:
    """Test Solvers factory class."""

    def test_get_adamw_solver(self):
        """Test getting AdamW solver."""
        solver = Solvers.get_solver("adamw")

        assert isinstance(solver, AdamWSolver)

    def test_adamw_default_parameters(self):
        """Test AdamW solver with default parameters."""
        solver = Solvers.get_solver("adamw")

        assert solver.beta_1 == 0.9
        assert solver.beta_2 == 0.999
        assert solver.epsilon == 1e-8
        assert solver.weight_decay == 1e-2

    @pytest.mark.parametrize("beta_1,beta_2,epsilon,weight_decay", [
        (0.9, 0.999, 1e-8, 1e-2),
        (0.8, 0.99, 1e-7, 1e-3),
        (0.95, 0.9999, 1e-9, 0.0),
    ])
    def test_adamw_with_custom_parameters(self, beta_1, beta_2, epsilon, weight_decay):
        """Test AdamW solver with custom parameters."""
        solver = Solvers.get_solver(
            "adamw",
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay
        )

        assert isinstance(solver, AdamWSolver)
        assert solver.beta_1 == beta_1
        assert solver.beta_2 == beta_2
        assert solver.epsilon == epsilon
        assert solver.weight_decay == weight_decay

    def test_invalid_solver_raises_keyerror(self):
        """Test that invalid solver name raises KeyError."""
        with pytest.raises(KeyError):
            Solvers.get_solver("invalid_solver")

    def test_solver_map_contains_expected_solvers(self):
        """Test that SOLVER_MAP contains expected solvers."""
        expected_keys = {"adamw"}
        actual_keys = set(Solvers.SOLVER_MAP.keys())

        assert expected_keys == actual_keys

    def test_solver_functionality(self):
        """Test that returned solver actually works."""
        solver = Solvers.get_solver("adamw")

        # Test compute_update_value method
        layer_id = 0
        learning_rate = 0.001
        gradient = np.array([[1.0, 2.0]])
        weights = np.array([[0.5, 0.6]])

        update = solver.compute_update_value(layer_id, learning_rate, gradient, weights)

        assert update.shape == weights.shape
        assert np.isfinite(update).all()


class TestLossesFactory:
    """Test Losses factory class."""

    @pytest.mark.parametrize("loss_name,expected_class", [
        ("mse", MSE),
        ("cross_entropy_multiclass", CrossEntropy_MultiClass),
    ])
    def test_get_loss_returns_correct_type(self, loss_name, expected_class):
        """Test that get_loss returns correct loss class."""
        loss = Losses.get_loss(loss_name)

        assert isinstance(loss, expected_class)

    def test_mse_loss_instance(self):
        """Test getting MSE loss instance."""
        loss = Losses.get_loss("mse")

        assert isinstance(loss, MSE)

        # Test that it works
        y = np.array([[1.0, 0.0]])
        y_hat = np.array([[0.9, 0.1]])
        result = loss.forward(y, y_hat)

        assert result >= 0
        assert np.isfinite(result)

    def test_cross_entropy_loss_instance(self):
        """Test getting cross-entropy loss instance."""
        loss = Losses.get_loss("cross_entropy_multiclass")

        assert isinstance(loss, CrossEntropy_MultiClass)

        # Test default epsilon
        assert loss.epsilon == 1e-7

    @pytest.mark.parametrize("epsilon", [1e-7, 1e-8, 1e-10])
    def test_cross_entropy_with_custom_epsilon(self, epsilon):
        """Test getting cross-entropy with custom epsilon parameter."""
        loss = Losses.get_loss("cross_entropy_multiclass", epsilon=epsilon)

        assert isinstance(loss, CrossEntropy_MultiClass)
        assert loss.epsilon == epsilon

    def test_invalid_loss_raises_keyerror(self):
        """Test that invalid loss name raises KeyError."""
        with pytest.raises(KeyError):
            Losses.get_loss("invalid_loss")

    def test_loss_map_contains_expected_losses(self):
        """Test that LOSS_MAP contains expected losses."""
        expected_keys = {"mse", "cross_entropy_multiclass"}
        actual_keys = set(Losses.LOSS_MAP.keys())

        assert expected_keys == actual_keys


class TestFactoryIntegration:
    """Integration tests for all factory classes."""

    def test_all_factories_work_together(self):
        """Test that all factories can be used together."""
        # Get components from factories
        activation = Activations.get_activation("leaky_relu")
        scheduler = LearningRates.get_learning_rate(
            "warmup_cosine_annealing",
            lr_min=1e-3,
            lr_max=0.1
        )
        solver = Solvers.get_solver("adamw")
        loss = Losses.get_loss("cross_entropy_multiclass")

        # Verify all are correct types
        assert isinstance(activation, LeakyReLUActivation)
        assert isinstance(scheduler, WarmUpCosineAnnealing)
        assert isinstance(solver, AdamWSolver)
        assert isinstance(loss, CrossEntropy_MultiClass)

    def test_factories_produce_independent_instances(self):
        """Test that factories produce independent instances."""
        # Get two solvers
        solver1 = Solvers.get_solver("adamw")
        solver2 = Solvers.get_solver("adamw")

        # They should be different objects
        assert solver1 is not solver2

        # Modifying one shouldn't affect the other
        solver1.beta_1 = 0.5
        assert solver2.beta_1 == 0.9  # Still default value

    def test_factory_pattern_benefits(self):
        """Test benefits of factory pattern - easy instantiation."""
        # Easy to create different configurations
        configs = [
            {"activation": "sigmoid"},
            {"activation": "leaky_relu", "alpha": 0.1},
            {"activation": "softmax"},
        ]

        activations = []
        for config in configs:
            act_name = config.pop("activation")
            activation = Activations.get_activation(act_name, **config)
            activations.append(activation)

        assert len(activations) == 3
        assert isinstance(activations[0], SigmoidActivation)
        assert isinstance(activations[1], LeakyReLUActivation)
        assert isinstance(activations[2], SoftmaxActivation)

    def test_all_activation_types_available(self):
        """Test that all activation types can be instantiated."""
        activation_names = ["sigmoid", "leaky_relu", "identity", "softmax"]

        for name in activation_names:
            activation = Activations.get_activation(name)
            assert activation is not None

            # Test basic functionality
            X = np.array([[1.0, 2.0]])
            result = activation.forward(X)
            assert result.shape[0] == 1  # Batch size preserved

    def test_all_loss_types_available(self):
        """Test that all loss types can be instantiated."""
        loss_names = ["mse", "cross_entropy_multiclass"]

        for name in loss_names:
            loss = Losses.get_loss(name)
            assert loss is not None

            # Test basic functionality
            y = np.array([[1.0, 0.0]])
            y_hat = np.array([[0.8, 0.2]])
            result = loss.forward(y, y_hat)
            assert np.isfinite(result)


class TestFactoryErrorHandling:
    """Test error handling in factory classes."""

    @pytest.mark.parametrize("factory_class,method_name,invalid_name", [
        (Activations, "get_activation", "relu"),
        (Activations, "get_activation", "tanh"),
        (LearningRates, "get_learning_rate", "step"),
        (LearningRates, "get_learning_rate", "exponential"),
        (Solvers, "get_solver", "sgd"),
        (Solvers, "get_solver", "adam"),
        (Losses, "get_loss", "binary_crossentropy"),
        (Losses, "get_loss", "mae"),
    ])
    def test_invalid_names_raise_keyerror(self, factory_class, method_name, invalid_name):
        """Test that invalid names raise KeyError for all factories."""
        method = getattr(factory_class, method_name)

        with pytest.raises(KeyError):
            method(invalid_name)

    def test_activation_with_invalid_kwargs(self):
        """Test activation creation with invalid keyword arguments."""
        # This should work - extra kwargs are ignored by most classes
        # or raise TypeError if strict
        try:
            activation = Activations.get_activation("sigmoid", invalid_param=123)
            # If it doesn't raise, that's also acceptable behavior
        except TypeError:
            # Expected if the class doesn't accept extra kwargs
            pass

    def test_case_sensitive_factory_names(self):
        """Test that factory names are case-sensitive."""
        with pytest.raises(KeyError):
            Activations.get_activation("Sigmoid")  # Capital S

        with pytest.raises(KeyError):
            Losses.get_loss("MSE")  # Capital letters

        with pytest.raises(KeyError):
            Solvers.get_solver("AdamW")  # Capital letters
