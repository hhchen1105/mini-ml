import pytest
import numpy as np
from numpy.testing import assert_array_equal
from miniml.sklearn.model_selection.StratifiedKFold import StratifiedKFold


def test_stratifiedkfold_get_n_splits():
    """Test that get_n_splits returns the correct number of splits"""
    skf = StratifiedKFold(n_splits=5)
    assert skf.get_n_splits() == 5

    skf = StratifiedKFold(n_splits=3)
    assert skf.get_n_splits() == 3

    # Test with X and y parameters (should be ignored)
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    assert skf.get_n_splits(X, y) == 3


def test_stratifiedkfold_split_basic_and_coverage():
    """Test basic split functionality and full coverage of indices"""
    X = np.arange(12).reshape(12, 1)
    y = np.array([0] * 6 + [1] * 6)  # balanced

    skf = StratifiedKFold(n_splits=3, shuffle=False)
    splits = list(skf.split(X, y))

    # Should have 3 splits
    assert len(splits) == 3

    # Check that each sample appears exactly once in the test sets
    all_test = np.concatenate([te for _, te in splits])
    assert_array_equal(np.sort(all_test), np.arange(len(X)))

    # Check that train and test indices are disjoint for each split
    for tr, te in splits:
        assert len(np.intersect1d(tr, te)) == 0


def test_stratifiedkfold_preserves_class_distribution_balanced():
    """Test that class distribution is preserved for balanced classes"""
    X = np.arange(20).reshape(20, 1)
    y = np.array([0] * 10 + [1] * 10)  # 50/50 overall

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for tr, te in skf.split(X, y):
        # Each test fold should have 2 samples from each class
        test_counts = np.bincount(y[te])
        assert_array_equal(test_counts, np.array([2, 2]))

        # Each train fold should have 8 samples from each class
        train_counts = np.bincount(y[tr])
        assert_array_equal(train_counts, np.array([8, 8]))


def test_stratifiedkfold_preserves_class_distribution_imbalanced():
    """Test that class distribution is preserved for imbalanced classes"""
    X = np.arange(30).reshape(30, 1)
    # 20 samples of class 0, 10 samples of class 1 => ratio 2:1
    y = np.array([0] * 20 + [1] * 10)

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for _, te in skf.split(X, y):
        # Each test fold has 6 samples => expected about 4 from class 0 and 2 from class 1
        test_counts = np.bincount(y[te])
        assert_array_equal(test_counts, np.array([4, 2]))


def test_stratifiedkfold_uneven_class_counts():
    """Test splits when class counts are uneven"""
    X = np.arange(14).reshape(14, 1)
    # class 0: 9 samples, class 1: 5 samples
    y = np.array([0] * 9 + [1] * 5)

    skf = StratifiedKFold(n_splits=3, shuffle=False)
    splits = list(skf.split(X, y))

    # Total test sizes across folds should sum to n_samples
    test_sizes = [len(te) for _, te in splits]
    assert sum(test_sizes) == len(X)

    # Each fold should roughly preserve the overall ratio
    for _, te in splits:
        test_counts = np.bincount(y[te], minlength=2)
        # Overall ratio 9:5 -> per fold around 3:2 or 3:1 depending on split
        assert test_counts[0] in (2, 3, 4)
        assert test_counts[1] in (1, 2)


def test_stratifiedkfold_shuffle_reproducible():
    """Test that shuffling with the same random_state is reproducible"""
    X = np.arange(12).reshape(12, 1)
    y = np.array([0] * 6 + [1] * 6)

    skf1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    skf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

    splits1 = list(skf1.split(X, y))
    splits2 = list(skf2.split(X, y))

    for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
        assert_array_equal(tr1, tr2)
        assert_array_equal(te1, te2)


def test_stratifiedkfold_requires_y():
    """Test that providing y is mandatory"""
    X = np.arange(6).reshape(6, 1)
    skf = StratifiedKFold(n_splits=3)

    with pytest.raises(ValueError, match="y must be provided"):
        list(skf.split(X, None))


def test_stratifiedkfold_invalid_n_splits_low():
    """Test that n_splits < 2 raises ValueError"""
    with pytest.raises(ValueError, match="must be at least 2"):
        StratifiedKFold(n_splits=1)


def test_stratifiedkfold_invalid_n_splits_too_large_for_min_class():
    """Test that n_splits larger than smallest class size raises ValueError"""
    X = np.arange(6).reshape(6, 1)
    y = np.array([0, 0, 0, 0, 1, 1])  # min class size = 2

    skf = StratifiedKFold(n_splits=3)
    with pytest.raises(ValueError, match="cannot be greater than the smallest class size"):
        list(skf.split(X, y))


def test_stratifiedkfold_mismatched_lengths():
    """Test that mismatched lengths between X and y raise ValueError"""
    X = np.arange(5).reshape(5, 1)
    y = np.array([0, 1, 0])  # length mismatch

    skf = StratifiedKFold(n_splits=2)
    with pytest.raises(ValueError, match="same length"):
        list(skf.split(X, y))


def test_stratifiedkfold_all_samples_used_in_training():
    """Test that each sample appears in training exactly (n_splits - 1) times"""
    X = np.arange(15).reshape(15, 1)
    y = np.array([0] * 9 + [1] * 6)

    skf = StratifiedKFold(n_splits=3)
    train_counts = np.zeros(len(X))

    for tr, _ in skf.split(X, y):
        train_counts[tr] += 1

    assert_array_equal(train_counts, np.full(len(X), 2))


@pytest.mark.parametrize("n_splits, n0, n1", [
    (2, 8, 4),
    (3, 9, 6),
    (4, 12, 8),
])
def test_stratifiedkfold_parametrized(n_splits, n0, n1):
    """Parametrized test for multiple class-size / split configurations"""
    X = np.arange(n0 + n1).reshape(n0 + n1, 1)
    y = np.array([0] * n0 + [1] * n1)

    skf = StratifiedKFold(n_splits=n_splits)
    splits = list(skf.split(X, y))

    assert len(splits) == n_splits

    all_test = np.concatenate([te for _, te in splits])
    assert_array_equal(np.sort(all_test), np.arange(len(X)))

    # Each fold should contain both classes (when possible)
    for _, te in splits:
        test_counts = np.bincount(y[te], minlength=2)
        assert test_counts[0] > 0
        assert test_counts[1] > 0
