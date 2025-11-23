import numpy as np


class StratifiedKFold:
    """
    Stratified K-Folds cross-validator.

    Provides train/test indices to split data into train/test sets.
    This cross-validator preserves the class distribution in each fold
    according to the labels y.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError(f"n_splits={n_splits} must be at least 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):
        """
        Generate train/test indices with stratification based on y.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Input data. Only the number of samples is used.
        y : array-like of shape (n_samples,)
            Class labels used for stratification.
        groups : ignored
            Exists for API compatibility.

        Yields
        ------
        train_idx : ndarray
            Indices of training samples.
        test_idx : ndarray
            Indices of test samples.
        """
        if y is None:
            raise ValueError("y must be provided for StratifiedKFold.")

        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = len(X)
        if len(y) != n_samples:
            raise ValueError("X and y must have the same length.")

        rng = np.random.RandomState(self.random_state)
        all_idx = np.arange(n_samples)

        # Collect indices for each class
        unique_classes, y_inv = np.unique(y, return_inverse=True)
        per_class = [all_idx[y_inv == k] for k in range(len(unique_classes))]

        # Optionally shuffle indices inside each class
        if self.shuffle:
            for idx in per_class:
                rng.shuffle(idx)

        # Validate that each class has enough samples for n_splits
        min_count = min(len(idx) for idx in per_class)
        if self.n_splits > min_count:
            raise ValueError(
                f"n_splits={self.n_splits} cannot be greater than "
                f"the smallest class size={min_count}."
            )

        # Split each class' indices into n_splits chunks
        class_chunks = [np.array_split(idx, self.n_splits) for idx in per_class]

        # Build each fold's test set by combining the corresponding chunk
        for fold_id in range(self.n_splits):
            test_idx = np.concatenate([chunks[fold_id] for chunks in class_chunks])
            test_idx = np.sort(test_idx)

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            train_idx = all_idx[train_mask]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return the number of splitting iterations in the cross-validator.
        """
        return self.n_splits
