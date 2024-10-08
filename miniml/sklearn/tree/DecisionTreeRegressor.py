import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = self.build_tree(X, y)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree_) for x in X])

    def predict_sample(self, x, tree):
        if isinstance(tree, dict):
            feature, threshold = tree['feature'], tree['threshold']
            if x[feature] <= threshold:
                return self.predict_sample(x, tree['left'])
            else:
                return self.predict_sample(x, tree['right'])
        else:
            return tree

    def build_tree(self, X, y, depth=0):
        if len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)
        
        feature, threshold, mse = self.best_split(X, y)
        if feature is None:
            return np.mean(y)
        
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        left_tree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {'feature': feature, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def best_split(self, X, y):
        best_feature, best_threshold, best_mse = None, None, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
                if len(X_left) == 0 or len(X_right) == 0:
                    continue
                mse = (self.mse(y_left) * len(y_left) + self.mse(y_right) * len(y_right)) / len(y)
                if mse < best_mse:
                    best_feature, best_threshold, best_mse = feature, threshold, mse
        return best_feature, best_threshold, best_mse

    def split(self, X, y, feature, threshold):
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

    def mse(self, y):
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
