"""
Comprehensive pytest test suite for WarmUpCosineAnnealing class.

This test suite covers:
- Initialization and parameter settings
- Warm-up phase behavior
- Cosine annealing phase behavior
- Learning rate computation
- Parametrization for different configurations
- Edge cases
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from math import pi, cos
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Module.Learning_Rate_Scheduler import WarmUpCosineAnnealing


class TestWarmUpCosineAnnealingInitialization:
    """Test WarmUpCosineAnnealing initialization."""

    def test_default_initialization(self):
        """Test scheduler initialization with default warm_up."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1)

        assert scheduler.lr_min == 1e-3
        assert scheduler.lr_max == 0.1
        assert scheduler.warm_up == 10

    @pytest.mark.parametrize("lr_min,lr_max,warm_up", [
        (1e-4, 0.1, 5),
        (1e-3, 0.05, 10),
        (1e-5, 0.5, 20),
        (0.0, 1.0, 0),
    ])
    def test_custom_initialization(self, lr_min, lr_max, warm_up):
        """Test scheduler initialization with custom parameters."""
        scheduler = WarmUpCosineAnnealing(lr_min=lr_min, lr_max=lr_max, warm_up=warm_up)

        assert scheduler.lr_min == lr_min
        assert scheduler.lr_max == lr_max
        assert scheduler.warm_up == warm_up


class TestWarmUpPhase:
    """Test warm-up phase behavior."""

    def test_warmup_first_iteration(self):
        """Test learning rate at first iteration during warm-up."""
        scheduler = WarmUpCosineAnnealing(lr_min=0.0, lr_max=0.1, warm_up=10)

        lr = scheduler.compute_lr(current_iter=0, max_iter=100, current_lr=0.0)

        # At iteration 0: lr = lr_max / warm_up * (0 + 1) = 0.1 / 10 = 0.01
        expected_lr = 0.01
        assert_allclose(lr, expected_lr, rtol=1e-6)

    @pytest.mark.parametrize("iteration,expected_fraction", [
        (0, 1/10),
        (1, 2/10),
        (4, 5/10),
        (9, 10/10),
    ])
    def test_warmup_linear_increase(self, iteration, expected_fraction):
        """Test that warm-up linearly increases learning rate."""
        scheduler = WarmUpCosineAnnealing(lr_min=0.0, lr_max=0.1, warm_up=10)

        lr = scheduler.compute_lr(current_iter=iteration, max_iter=100, current_lr=0.0)

        expected_lr = 0.1 * expected_fraction
        assert_allclose(lr, expected_lr, rtol=1e-6)

    def test_warmup_reaches_max_lr(self):
        """Test that warm-up reaches lr_max at the end."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=5)

        lr = scheduler.compute_lr(current_iter=4, max_iter=100, current_lr=0.0)

        # At iteration 4 (last warm-up): lr = lr_max / 5 * (4 + 1) = 0.1
        assert_allclose(lr, 0.1, rtol=1e-6)

    @pytest.mark.parametrize("warm_up", [1, 5, 10, 20])
    def test_warmup_different_periods(self, warm_up):
        """Test warm-up with different warm-up periods."""
        scheduler = WarmUpCosineAnnealing(lr_min=0.0, lr_max=0.1, warm_up=warm_up)

        # At last warm-up iteration
        lr = scheduler.compute_lr(current_iter=warm_up-1, max_iter=100, current_lr=0.0)

        # Should reach lr_max
        assert_allclose(lr, 0.1, rtol=1e-6)


class TestCosineAnnealingPhase:
    """Test cosine annealing phase behavior."""

    def test_annealing_starts_after_warmup(self):
        """Test that cosine annealing starts after warm-up."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=10)

        # At iteration 10 (first annealing iteration, adjusted iter = 0)
        lr = scheduler.compute_lr(current_iter=10, max_iter=100, current_lr=0.0)

        # At annealing iter 0: lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(0))
        # = 1e-3 + (0.1 - 1e-3) * 0.5 * 2 = 1e-3 + 0.099 = 0.1
        expected_lr = 1e-3 + (0.1 - 1e-3) * 0.5 * (1 + cos(0))
        assert_allclose(lr, expected_lr, rtol=1e-6)

    def test_annealing_ends_at_min_lr(self):
        """Test that cosine annealing ends at lr_min."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=10)
        max_iter = 100

        # At last iteration (iteration 99, annealing iter = 89)
        lr = scheduler.compute_lr(current_iter=99, max_iter=max_iter, current_lr=0.0)

        # At annealing iter 89 (max annealing iter = 90):
        # lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(89 * pi / 90))
        # cos(89*pi/90) ≈ cos(pi) = -1, so lr ≈ lr_min
        assert lr < 0.05  # Should be close to lr_min

    @pytest.mark.parametrize("current_iter,max_iter,warm_up", [
        (50, 100, 10),
        (30, 50, 5),
        (70, 100, 20),
    ])
    def test_annealing_monotonic_decrease(self, current_iter, max_iter, warm_up):
        """Test that cosine annealing generally decreases over time."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=warm_up)

        if current_iter < warm_up:
            pytest.skip("Test only for annealing phase")

        lr_current = scheduler.compute_lr(current_iter=current_iter, max_iter=max_iter, current_lr=0.0)
        lr_next = scheduler.compute_lr(current_iter=current_iter+1, max_iter=max_iter, current_lr=0.0)

        # Learning rate should decrease during annealing
        assert lr_next <= lr_current

    def test_annealing_cosine_formula(self):
        """Test that cosine annealing follows correct formula."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=10)
        max_iter = 100
        current_iter = 50

        lr = scheduler.compute_lr(current_iter=current_iter, max_iter=max_iter, current_lr=0.0)

        # Manual calculation
        adjusted_max_iter = max_iter - 10  # 90
        adjusted_current_iter = current_iter - 10  # 40
        expected_lr = 1e-3 + (0.1 - 1e-3) * 0.5 * (1 + cos(adjusted_current_iter * pi / adjusted_max_iter))

        assert_allclose(lr, expected_lr, rtol=1e-6)


class TestComputeLR:
    """Test compute_lr method in various scenarios."""

    def test_compute_lr_returns_positive(self):
        """Test that compute_lr always returns positive values."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=10)

        for iter in range(100):
            lr = scheduler.compute_lr(current_iter=iter, max_iter=100, current_lr=0.0)
            assert lr > 0

    def test_compute_lr_stays_in_bounds(self):
        """Test that compute_lr stays within [lr_min, lr_max]."""
        lr_min = 1e-3
        lr_max = 0.1
        scheduler = WarmUpCosineAnnealing(lr_min=lr_min, lr_max=lr_max, warm_up=10)

        for iter in range(100):
            lr = scheduler.compute_lr(current_iter=iter, max_iter=100, current_lr=0.0)
            assert lr >= lr_min - 1e-6  # Allow small numerical error
            assert lr <= lr_max + 1e-6

    @pytest.mark.parametrize("lr_min,lr_max", [
        (1e-4, 0.1),
        (1e-3, 0.05),
        (1e-5, 1.0),
    ])
    def test_compute_lr_different_lr_ranges(self, lr_min, lr_max):
        """Test compute_lr with different learning rate ranges."""
        scheduler = WarmUpCosineAnnealing(lr_min=lr_min, lr_max=lr_max, warm_up=10)

        # During warm-up
        lr_warmup = scheduler.compute_lr(current_iter=5, max_iter=100, current_lr=0.0)
        assert lr_warmup <= lr_max

        # During annealing
        lr_annealing = scheduler.compute_lr(current_iter=50, max_iter=100, current_lr=0.0)
        assert lr_min <= lr_annealing <= lr_max

    def test_current_lr_parameter_unused(self):
        """Test that current_lr parameter doesn't affect result."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=10)

        lr1 = scheduler.compute_lr(current_iter=20, max_iter=100, current_lr=0.0)
        lr2 = scheduler.compute_lr(current_iter=20, max_iter=100, current_lr=0.05)
        lr3 = scheduler.compute_lr(current_iter=20, max_iter=100, current_lr=0.2)

        # All should return the same value (current_lr is not used in the implementation)
        assert_allclose(lr1, lr2, rtol=1e-6)
        assert_allclose(lr2, lr3, rtol=1e-6)


class TestWarmUpCosineAnnealingEdgeCases:
    """Test edge cases for WarmUpCosineAnnealing."""

    def test_zero_warmup(self):
        """Test scheduler with zero warm-up iterations."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=0)

        # First iteration should start cosine annealing immediately
        lr = scheduler.compute_lr(current_iter=0, max_iter=100, current_lr=0.0)

        # cos(0) = 1, so lr = lr_min + (lr_max - lr_min) * 0.5 * 2 = lr_max
        expected_lr = 1e-3 + (0.1 - 1e-3) * 0.5 * (1 + cos(0))
        assert_allclose(lr, expected_lr, rtol=1e-6)

    def test_warmup_equals_max_iter(self):
        """Test scheduler when warm-up equals max iterations."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=100)

        # All iterations are warm-up
        lr_last = scheduler.compute_lr(current_iter=99, max_iter=100, current_lr=0.0)

        # Should reach lr_max
        assert_allclose(lr_last, 0.1, rtol=1e-6)

    def test_single_iteration(self):
        """Test scheduler with single total iteration."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=1)

        lr = scheduler.compute_lr(current_iter=0, max_iter=1, current_lr=0.0)

        # Should be in warm-up phase
        assert lr > 0

    def test_lr_min_equals_lr_max(self):
        """Test scheduler when lr_min equals lr_max."""
        scheduler = WarmUpCosineAnnealing(lr_min=0.1, lr_max=0.1, warm_up=10)

        for iter in range(100):
            lr = scheduler.compute_lr(current_iter=iter, max_iter=100, current_lr=0.0)
            # Should always be lr_max (= lr_min)
            assert_allclose(lr, scheduler.lr_max, rtol=1e-6)


class TestWarmUpCosineAnnealingIntegration:
    """Integration tests for WarmUpCosineAnnealing."""

    def test_full_schedule_progression(self):
        """Test complete learning rate schedule from start to end."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=10)
        max_iter = 100

        learning_rates = []
        for iter in range(max_iter):
            lr = scheduler.compute_lr(current_iter=iter, max_iter=max_iter, current_lr=0.0)
            learning_rates.append(lr)

        # Check warm-up phase: increasing
        for i in range(9):
            assert learning_rates[i] < learning_rates[i+1], f"Warm-up should increase at iter {i}"

        # Check that lr_max is reached around end of warm-up
        assert_allclose(learning_rates[9], 0.1, rtol=1e-6)

        # Check annealing phase: generally decreasing
        # (some tolerance for numerical issues)
        for i in range(10, max_iter - 1):
            assert learning_rates[i] >= learning_rates[i+1] - 1e-6, f"Annealing should decrease at iter {i}"

        # Check final lr is close to lr_min
        assert learning_rates[-1] < 0.05

    def test_multiple_complete_schedules(self):
        """Test running multiple complete schedules."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=5)
        max_iter = 50

        # First schedule
        lr_schedule_1 = [scheduler.compute_lr(i, max_iter, 0.0) for i in range(max_iter)]

        # Second schedule (should be identical)
        lr_schedule_2 = [scheduler.compute_lr(i, max_iter, 0.0) for i in range(max_iter)]

        # Schedules should be identical (stateless)
        assert_allclose(lr_schedule_1, lr_schedule_2, rtol=1e-10)

    def test_different_max_iters_same_scheduler(self):
        """Test that max_iter parameter affects the schedule correctly."""
        scheduler = WarmUpCosineAnnealing(lr_min=1e-3, lr_max=0.1, warm_up=10)

        # With max_iter=50, at iter 30
        lr_50 = scheduler.compute_lr(current_iter=30, max_iter=50, current_lr=0.0)

        # With max_iter=100, at iter 30
        lr_100 = scheduler.compute_lr(current_iter=30, max_iter=100, current_lr=0.0)

        # Different max_iter should give different learning rates
        assert not np.isclose(lr_50, lr_100)
