import numpy as np

class ShuffleSplit:
    def __init__(self, n_splits=10, test_size=0.1, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X):
        n_samples = X.shape[0]
        n_test = int(n_samples * self.test_size)
        n_train = int(n_samples * self.train_size) if self.train_size is not None else n_samples - n_test
        rng = np.random.default_rng(self.random_state)
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if n_test <= 0 or n_test >= n_samples:
            raise ValueError("invalid test_size")
        if n_train < 0 or n_train >= n_samples:
            raise ValueError("invalid train_size")
        for _ in range(self.n_splits):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
            yield train_indices, test_indices