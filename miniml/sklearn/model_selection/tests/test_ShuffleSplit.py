import pytest
import numpy as np
from miniml.sklearn.model_selection.ShuffleSplit import ShuffleSplit

@pytest.mark.parametrize("n, test_size", [
    (100, 0.2), (100, 0.33), (37, 0.2),
(50, 0.1),# int
])

def test_indices_range_and_lengths(n, test_size):
    X = np.arange(n)
    ss = ShuffleSplit(n_splits=3, test_size=test_size, random_state=0)
    for train_idx, test_idx in ss.split(X):
        assert train_idx.ndim == 1 and test_idx.ndim == 1
        assert train_idx.dtype.kind in "iu" and test_idx.dtype.kind in "iu"
        # 範圍
        assert np.all(train_idx >= 0) and np.all(train_idx < n)
        assert np.all(test_idx >= 0) and np.all(test_idx < n)
        # 互斥 & 不重複
        assert len(set(train_idx)) == len(train_idx)
        assert len(set(test_idx)) == len(test_idx)
        assert set(train_idx).isdisjoint(set(test_idx))
        # 長度
        if isinstance(test_size, float):
            expected_test = int(round(test_size * n))
        else:
            expected_test = test_size if isinstance(test_size, int) else int(0.2*n)
        assert len(test_idx) == expected_test
        assert len(train_idx) == n - expected_test

def test_random_state_reproducible():
    X = np.arange(50)
    ss1 = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    ss2 = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    out1 = list(ss1.split(X))
    out2 = list(ss2.split(X))
    assert all(
        np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        for a, b in zip(out1, out2)
    )

def test_random_state_different_seeds():
    X = np.arange(50)
    a = list(ShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(X))[0]
    b = list(ShuffleSplit(n_splits=1, test_size=0.2, random_state=2).split(X))[0]
    # 高機率不同
    assert not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))


if __name__ == "__main__":
    pytest.main()