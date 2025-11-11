import pytest
import numpy as np
from numpy.testing import assert_array_equal
from miniml.sklearn.model_selection.KFold import KFold


def test_kfold_get_n_splits():
    """Test that get_n_splits returns the correct number of splits"""
    kf = KFold(n_splits=5)
    assert kf.get_n_splits() == 5
    
    kf = KFold(n_splits=3)
    assert kf.get_n_splits() == 3
    
    # Test with X parameter (should be ignored)
    X = np.array([[1, 2], [3, 4], [5, 6]])
    assert kf.get_n_splits(X) == 3


def test_kfold_split_basic():
    """Test basic split functionality"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    kf = KFold(n_splits=3)
    
    splits = list(kf.split(X))
    
    # Should have 3 splits
    assert len(splits) == 3
    
    # Check that each sample appears exactly once in test set
    all_test_indices = np.concatenate([test for _, test in splits])
    assert_array_equal(np.sort(all_test_indices), np.arange(len(X)))
    
    # Check that train and test are disjoint for each split
    for train, test in splits:
        assert len(np.intersect1d(train, test)) == 0


def test_kfold_split_sizes():
    """Test that split sizes are correct"""
    X = np.arange(10).reshape(10, 1)
    kf = KFold(n_splits=5)
    
    for train, test in kf.split(X):
        # Each test fold should have 2 samples
        assert len(test) == 2
        # Each train fold should have 8 samples
        assert len(train) == 8


def test_kfold_split_uneven():
    """Test split with uneven number of samples"""
    X = np.arange(11).reshape(11, 1)
    kf = KFold(n_splits=3)
    
    splits = list(kf.split(X))
    test_sizes = [len(test) for _, test in splits]
    
    # With 11 samples and 3 folds: sizes should be [4, 4, 3]
    assert test_sizes == [4, 4, 3]


def test_kfold_shuffle():
    """Test that shuffle parameter works"""
    X = np.arange(10).reshape(10, 1)
    
    # Without shuffle, splits should be deterministic
    kf1 = KFold(n_splits=5, shuffle=False)
    splits1 = list(kf1.split(X))
    
    kf2 = KFold(n_splits=5, shuffle=False)
    splits2 = list(kf2.split(X))
    
    # Same results without shuffle
    for (train1, test1), (train2, test2) in zip(splits1, splits2):
        assert_array_equal(train1, train2)
        assert_array_equal(test1, test2)
    
    # With shuffle and random_state, should be reproducible
    kf3 = KFold(n_splits=5, shuffle=True, random_state=42)
    splits3 = list(kf3.split(X))
    
    kf4 = KFold(n_splits=5, shuffle=True, random_state=42)
    splits4 = list(kf4.split(X))
    
    # Same results with same random_state
    for (train3, test3), (train4, test4) in zip(splits3, splits4):
        assert_array_equal(train3, train4)
        assert_array_equal(test3, test4)


def test_kfold_with_y():
    """Test that y parameter is accepted (even if ignored)"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    
    kf = KFold(n_splits=2)
    splits = list(kf.split(X, y))
    
    assert len(splits) == 2


def test_kfold_invalid_n_splits():
    """Test that invalid n_splits raises ValueError"""
    with pytest.raises(ValueError, match="n_splits=1 must be at least 2"):
        KFold(n_splits=1)
    
    with pytest.raises(ValueError, match="n_splits=0 must be at least 2"):
        KFold(n_splits=0)


def test_kfold_all_samples_used():
    """Test that all samples are used in training across all folds"""
    X = np.arange(20).reshape(20, 1)
    kf = KFold(n_splits=4)
    
    # Track how many times each sample appears in training
    train_counts = np.zeros(len(X))
    
    for train, test in kf.split(X):
        train_counts[train] += 1
    
    # Each sample should appear in training exactly (n_splits - 1) times
    assert_array_equal(train_counts, np.full(len(X), 3))


@pytest.mark.parametrize("n_splits,n_samples", [
    (2, 10),
    (3, 9),
    (5, 15),
    (4, 8),
])
def test_kfold_parametrized(n_splits, n_samples):
    """Parametrized test for different combinations"""
    X = np.arange(n_samples).reshape(n_samples, 1)
    kf = KFold(n_splits=n_splits)
    
    assert kf.get_n_splits(X) == n_splits
    
    splits = list(kf.split(X))
    assert len(splits) == n_splits
    
    # Verify all indices are covered
    all_test = np.concatenate([test for _, test in splits])
    assert_array_equal(np.sort(all_test), np.arange(n_samples))
