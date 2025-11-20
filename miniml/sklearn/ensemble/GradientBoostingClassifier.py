import numpy as np
from ..tree.DecisionTreeRegressor import DecisionTreeRegressor


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        """
        Gradient Boosting Classifier

        Parameters:
        -----------
        n_estimators : int, default=100
            The number of boosting stages to perform
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree
        max_depth : int, default=3
            Maximum depth of the individual trees
        random_state : int, default=None
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.init_prediction = None
        self.classes_ = None
        self.n_features_in_ = None

    def _sigmoid(self, x):
        """Sigmoid function for converting log-odds to probabilities"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _validate_hyperparameters(self):
        """Validate hyperparameters"""
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError(f"n_estimators must be an integer >= 1, got {self.n_estimators}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth < 1):
            raise ValueError(f"max_depth must be a positive integer or None, got {self.max_depth}")

    def _validate_input(self, X, y=None, is_fit=True):
        """Validate input data"""
        # Convert to numpy array
        X = np.asarray(X)

        # Check X is 2D
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D array with shape {X.shape}")

        if is_fit:
            # Validate y
            if y is None:
                raise ValueError("y cannot be None during fit")

            y = np.asarray(y)

            # Check y is 1D
            if y.ndim != 1:
                raise ValueError(f"y must be 1D array, got {y.ndim}D array with shape {y.shape}")

            # Check X and y have same number of samples
            if X.shape[0] != len(y):
                raise ValueError(f"X and y must have same number of samples. X: {X.shape[0]}, y: {len(y)}")

            # Check binary labels
            unique_labels = np.unique(y)

            # Accept {0, 1} or {-1, 1}
            if set(unique_labels) == {0, 1}:
                # Convert to {0, 1} (already in correct format)
                y_binary = y.copy()
            elif set(unique_labels) == {-1, 1}:
                # Convert {-1, 1} to {0, 1}
                y_binary = ((y + 1) / 2).astype(int)
            elif len(unique_labels) == 1:
                if unique_labels[0] in [0, 1, -1]:
                    # Only one class present
                    raise ValueError(f"y contains only one class: {unique_labels[0]}. Need at least 2 classes.")
                else:
                    raise ValueError(f"y must contain binary labels in {{0, 1}} or {{-1, 1}}, got {unique_labels}")
            else:
                raise ValueError(f"y must contain binary labels in {{0, 1}} or {{-1, 1}}, got {unique_labels}")

            return X, y_binary
        else:
            # Prediction: check feature dimensions
            if self.n_features_in_ is None:
                raise ValueError("This GradientBoostingClassifier instance is not fitted yet")

            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"X has {X.shape[1]} features, but GradientBoostingClassifier is expecting {self.n_features_in_} features")

            return X

    def fit(self, X, y):
        """
        Fit the gradient boosting classifier

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1, or -1 and 1)

        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Validate hyperparameters
        self._validate_hyperparameters()

        # Validate and convert input
        X, y = self._validate_input(X, y, is_fit=True)

        # Store original classes and number of features
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        # Reset trees
        self.trees = []

        # Create local random state instead of polluting global
        rng = np.random.RandomState(self.random_state)

        # Initialize with log-odds of the mean
        p = np.mean(y)
        p = np.clip(p, 1e-7, 1 - 1e-7)  # Avoid log(0)
        self.init_prediction = np.log(p / (1 - p))

        # Initialize predictions with log-odds
        F = np.full(len(y), self.init_prediction)

        # Build trees sequentially
        for _ in range(self.n_estimators):
            # Calculate probabilities and residuals (gradient)
            p = self._sigmoid(F)
            residuals = y - p

            # Fit a tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Make predictions (these are residual predictions)
            predictions = tree.predict(X)

            # Update F with learning rate
            F += self.learning_rate * predictions

            # Store the tree
            self.trees.append(tree)

        return self

    def _predict_raw(self, X):
        """
        Predict raw values (log-odds) before sigmoid transformation

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns:
        --------
        raw_predictions : array of shape (n_samples,)
            Raw predictions (log-odds)
        """
        # Start with initial prediction
        F = np.full(X.shape[0], self.init_prediction)

        # Add contribution from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        return F

    def predict_proba(self, X):
        """
        Predict class probabilities

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns:
        --------
        proba : array of shape (n_samples, 2)
            Class probabilities
        """
        # Validate input
        X = self._validate_input(X, is_fit=False)

        raw_predictions = self._predict_raw(X)
        prob_class_1 = self._sigmoid(raw_predictions)
        prob_class_0 = 1 - prob_class_1
        return np.column_stack([prob_class_0, prob_class_1])

    def predict(self, X):
        """
        Predict class labels

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns:
        --------
        predictions : array of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        # Validate input
        X = self._validate_input(X, is_fit=False)

        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def score(self, X, y):
        """
        Calculate accuracy score

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like of shape (n_samples,)
            True labels

        Returns:
        --------
        score : float
            Accuracy score
        """
        # Validate input
        X = self._validate_input(X, is_fit=False)
        y = np.asarray(y)

        # Convert {-1, 1} to {0, 1} if needed
        unique_y = np.unique(y)
        if set(unique_y) == {-1, 1}:
            y = ((y + 1) / 2).astype(int)

        return np.mean(self.predict(X) == y)
