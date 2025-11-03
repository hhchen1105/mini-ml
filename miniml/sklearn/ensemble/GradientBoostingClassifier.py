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

    def _sigmoid(self, x):
        """Sigmoid function for converting log-odds to probabilities"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        """
        Fit the gradient boosting classifier

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1)
        """
        np.random.seed(self.random_state)

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

            # Store the tree (we need to modify tree to work with residuals)
            # For simplicity, we'll store a regression-like tree
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
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

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
        return np.mean(self.predict(X) == y)
