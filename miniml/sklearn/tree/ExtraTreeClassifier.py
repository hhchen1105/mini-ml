import numpy as np

class ExtraTreeClassifier:
    """A simplified ExtraTreeClassifier for the mini-ml project.
    
    Methods to implement:
    - fit(X, y)
    - predict(X)
    - predict_proba(X)
    """

    def __init__(self, random_state=None):
        """Initialize classifier with optional random state."""
        self.random_state = random_state
        self.is_fitted_ = False

    def fit(self, X, y):
        """Fit the model on training data (to be implemented by member A)."""
        raise NotImplementedError("fit() method not yet implemented.")

    def predict(self, X):
        """Predict class labels (to be implemented by member B)."""
        raise NotImplementedError("predict() method not yet implemented.")

    def predict_proba(self, X):
        """Predict class probabilities (to be implemented by member C)."""
        raise NotImplementedError("predict_proba() method not yet implemented.")
