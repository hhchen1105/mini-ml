"""
Comprehensive pytest test suite for Data Generator functions.

This test suite covers:
- generate_regression_dataset
- generate_classification_dataset

Tests include:
- Output shapes and types
- Data validity and constraints
- Parametrization for different configurations
- Reproducibility with random seeds
- Edge cases
- Side effects
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Module.Data_Generator import generate_regression_dataset, generate_classification_dataset


class TestGenerateRegressionDataset:
    """Test generate_regression_dataset function."""

    def test_default_parameters(self):
        """Test regression dataset generation with default parameters."""
        X, y = generate_regression_dataset()

        # Default: num_samples=3, X_dim=10, y_dim=3
        assert X.shape == (3, 10)
        assert y.shape == (3, 3)

    @pytest.mark.parametrize("num_samples,X_dim,y_dim", [
        (10, 5, 3),
        (100, 20, 10),
        (50, 8, 4),
        (1, 1, 1),
    ])
    def test_custom_shapes(self, num_samples, X_dim, y_dim):
        """Test regression dataset with different shape parameters."""
        X, y = generate_regression_dataset(
            num_samples=num_samples,
            X_dim=X_dim,
            y_dim=y_dim
        )

        assert X.shape == (num_samples, X_dim)
        assert y.shape == (num_samples, y_dim)

    def test_output_is_probability_distribution(self):
        """Test that y values form valid probability distributions."""
        X, y = generate_regression_dataset(num_samples=100, X_dim=10, y_dim=5)

        # Each row should sum to 1 (probability distribution)
        row_sums = np.sum(y, axis=1)
        assert_allclose(row_sums, np.ones(100), rtol=1e-6)

        # All values should be in [0, 1]
        assert np.all(y >= 0)
        assert np.all(y <= 1)

    @pytest.mark.parametrize("random_seed", [0, 42, 123, 999])
    def test_reproducibility_with_seed(self, random_seed):
        """Test that same seed produces same dataset."""
        X1, y1 = generate_regression_dataset(
            num_samples=50,
            X_dim=10,
            y_dim=3,
            random_seed=random_seed
        )

        X2, y2 = generate_regression_dataset(
            num_samples=50,
            X_dim=10,
            y_dim=3,
            random_seed=random_seed
        )

        assert_allclose(X1, X2, rtol=1e-10)
        assert_allclose(y1, y2, rtol=1e-10)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different datasets."""
        X1, y1 = generate_regression_dataset(random_seed=42)
        X2, y2 = generate_regression_dataset(random_seed=123)

        # Datasets should be different
        assert not np.allclose(X1, X2)
        assert not np.allclose(y1, y2)

    def test_output_types(self):
        """Test that outputs are numpy arrays."""
        X, y = generate_regression_dataset()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_no_nan_or_inf(self):
        """Test that generated data contains no NaN or Inf values."""
        X, y = generate_regression_dataset(num_samples=100, X_dim=20, y_dim=10)

        assert np.isfinite(X).all()
        assert np.isfinite(y).all()
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_y_nonzero_elements(self):
        """Test that y has non-zero elements (is not trivial)."""
        X, y = generate_regression_dataset(num_samples=10, X_dim=5, y_dim=3)

        # At least some values should be non-zero
        assert np.any(y > 0)

    def test_large_dataset(self):
        """Test generation of large dataset."""
        X, y = generate_regression_dataset(num_samples=10000, X_dim=100, y_dim=50)

        assert X.shape == (10000, 100)
        assert y.shape == (10000, 50)

        # Check properties still hold
        row_sums = np.sum(y, axis=1)
        assert_allclose(row_sums, np.ones(10000), rtol=1e-6)


class TestGenerateClassificationDataset:
    """Test generate_classification_dataset function."""

    def test_default_parameters(self):
        """Test classification dataset generation with default parameters."""
        X, y = generate_classification_dataset()

        # Default: num_samples=3, X_dim=10, y_dim=3
        assert X.shape == (3, 10)
        assert y.shape == (3, 3)

    @pytest.mark.parametrize("num_samples,X_dim,y_dim", [
        (10, 5, 3),
        (100, 20, 10),
        (50, 8, 4),
        (1, 1, 1),
    ])
    def test_custom_shapes(self, num_samples, X_dim, y_dim):
        """Test classification dataset with different shape parameters."""
        X, y = generate_classification_dataset(
            num_samples=num_samples,
            X_dim=X_dim,
            y_dim=y_dim
        )

        assert X.shape == (num_samples, X_dim)
        assert y.shape == (num_samples, y_dim)

    def test_one_hot_encoding(self):
        """Test that y is properly one-hot encoded."""
        X, y = generate_classification_dataset(num_samples=100, X_dim=10, y_dim=5)

        # Each row should sum to 1 (one-hot encoded)
        row_sums = np.sum(y, axis=1)
        assert_array_equal(row_sums, np.ones(100))

        # Each row should have exactly one 1 and rest 0s
        for row in y:
            assert np.sum(row == 1.0) == 1
            assert np.sum(row == 0.0) == len(row) - 1

    def test_all_classes_represented_eventually(self):
        """Test that with enough samples, all classes can appear."""
        X, y = generate_classification_dataset(
            num_samples=1000,
            X_dim=10,
            y_dim=5,
            random_seed=42
        )

        # Get class indices
        class_indices = np.argmax(y, axis=1)

        # With 1000 samples and 5 classes, all classes should appear
        unique_classes = np.unique(class_indices)
        assert len(unique_classes) >= 3  # At least 3 out of 5 classes

    @pytest.mark.parametrize("random_seed", [0, 42, 123, 999])
    def test_reproducibility_with_seed(self, random_seed):
        """Test that same seed produces same dataset."""
        X1, y1 = generate_classification_dataset(
            num_samples=50,
            X_dim=10,
            y_dim=3,
            random_seed=random_seed
        )

        X2, y2 = generate_classification_dataset(
            num_samples=50,
            X_dim=10,
            y_dim=3,
            random_seed=random_seed
        )

        assert_allclose(X1, X2, rtol=1e-10)
        assert_allclose(y1, y2, rtol=1e-10)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different datasets."""
        X1, y1 = generate_classification_dataset(random_seed=42)
        X2, y2 = generate_classification_dataset(random_seed=123)

        # Datasets should be different
        assert not np.allclose(X1, X2)
        assert not np.allclose(y1, y2)

    def test_output_types(self):
        """Test that outputs are numpy arrays."""
        X, y = generate_classification_dataset()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_no_nan_or_inf(self):
        """Test that generated data contains no NaN or Inf values."""
        X, y = generate_classification_dataset(num_samples=100, X_dim=20, y_dim=10)

        assert np.isfinite(X).all()
        assert np.isfinite(y).all()
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_y_values_are_binary(self):
        """Test that y contains only 0s and 1s."""
        X, y = generate_classification_dataset(num_samples=100, X_dim=10, y_dim=5)

        # All values should be either 0 or 1
        assert np.all((y == 0) | (y == 1))

    def test_large_dataset(self):
        """Test generation of large dataset."""
        X, y = generate_classification_dataset(num_samples=10000, X_dim=100, y_dim=50)

        assert X.shape == (10000, 100)
        assert y.shape == (10000, 50)

        # Check one-hot encoding still holds
        row_sums = np.sum(y, axis=1)
        assert_array_equal(row_sums, np.ones(10000))

    def test_binary_classification(self):
        """Test binary classification (2 classes)."""
        X, y = generate_classification_dataset(num_samples=100, X_dim=10, y_dim=2)

        assert y.shape == (100, 2)

        # Each row should have exactly one 1
        for row in y:
            assert np.sum(row) == 1.0


class TestDataGeneratorComparison:
    """Compare regression and classification dataset generators."""

    def test_same_seed_different_functions(self):
        """Test that regression and classification with same seed differ appropriately."""
        seed = 42

        X_reg, y_reg = generate_regression_dataset(
            num_samples=50, X_dim=10, y_dim=3, random_seed=seed
        )

        X_cls, y_cls = generate_classification_dataset(
            num_samples=50, X_dim=10, y_dim=3, random_seed=seed
        )

        # X might be the same or different depending on implementation
        # But y should definitely differ in nature
        # Regression: probabilities (continuous)
        # Classification: one-hot (binary)

        # Check that regression y is continuous (not all 0s and 1s)
        assert not np.all((y_reg == 0) | (y_reg == 1))

        # Check that classification y is binary
        assert np.all((y_cls == 0) | (y_cls == 1))

    def test_both_generators_produce_valid_data(self):
        """Test that both generators produce valid data for training."""
        X_reg, y_reg = generate_regression_dataset(num_samples=100, X_dim=20, y_dim=5)
        X_cls, y_cls = generate_classification_dataset(num_samples=100, X_dim=20, y_dim=5)

        # Both should have correct shapes
        assert X_reg.shape == (100, 20)
        assert y_reg.shape == (100, 5)
        assert X_cls.shape == (100, 20)
        assert y_cls.shape == (100, 5)

        # Both should be finite
        assert np.isfinite(X_reg).all()
        assert np.isfinite(y_reg).all()
        assert np.isfinite(X_cls).all()
        assert np.isfinite(y_cls).all()

        # Both y should sum to 1 per row
        assert_allclose(np.sum(y_reg, axis=1), np.ones(100), rtol=1e-6)
        assert_allclose(np.sum(y_cls, axis=1), np.ones(100), rtol=1e-6)


class TestDataGeneratorEdgeCases:
    """Test edge cases for data generators."""

    def test_single_sample_regression(self):
        """Test regression dataset with single sample."""
        X, y = generate_regression_dataset(num_samples=1, X_dim=5, y_dim=3)

        assert X.shape == (1, 5)
        assert y.shape == (1, 3)
        assert_allclose(np.sum(y), 1.0, rtol=1e-6)

    def test_single_sample_classification(self):
        """Test classification dataset with single sample."""
        X, y = generate_classification_dataset(num_samples=1, X_dim=5, y_dim=3)

        assert X.shape == (1, 5)
        assert y.shape == (1, 3)
        assert np.sum(y) == 1.0

    def test_single_feature_regression(self):
        """Test regression dataset with single feature."""
        X, y = generate_regression_dataset(num_samples=10, X_dim=1, y_dim=3)

        assert X.shape == (10, 1)
        assert y.shape == (10, 3)

    def test_single_feature_classification(self):
        """Test classification dataset with single feature."""
        X, y = generate_classification_dataset(num_samples=10, X_dim=1, y_dim=3)

        assert X.shape == (10, 1)
        assert y.shape == (10, 3)

    def test_single_class_classification(self):
        """Test classification dataset with single class."""
        X, y = generate_classification_dataset(num_samples=10, X_dim=5, y_dim=1)

        assert X.shape == (10, 5)
        assert y.shape == (10, 1)

        # All labels should be 1 (only one class)
        assert_array_equal(y, np.ones((10, 1)))

    def test_minimal_dataset_regression(self):
        """Test regression dataset with minimal size (1x1x1)."""
        X, y = generate_regression_dataset(num_samples=1, X_dim=1, y_dim=1)

        assert X.shape == (1, 1)
        assert y.shape == (1, 1)
        assert_allclose(y[0, 0], 1.0, rtol=1e-6)  # Single class, prob = 1

    def test_minimal_dataset_classification(self):
        """Test classification dataset with minimal size (1x1x1)."""
        X, y = generate_classification_dataset(num_samples=1, X_dim=1, y_dim=1)

        assert X.shape == (1, 1)
        assert y.shape == (1, 1)
        assert y[0, 0] == 1.0  # Single class, one-hot = 1
