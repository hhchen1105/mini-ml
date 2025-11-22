import numpy as np


class RadiusNeighborsClassifier:
	def __init__(
		self,
		radius=1.0,
		*,
		weights="uniform",
		# Default to brute; other algorithms would need extra dependencies.
		algorithm="brute",
		p=2,
		metric="minkowski",
		outlier_label=None,
	):
		self.radius = radius
		self.weights = weights
		self.algorithm = algorithm
		self.p = p
		self.metric = metric
		self.outlier_label = outlier_label

	def fit(self, X, y):
		X = np.asarray(X)
		y = np.asarray(y)

		if X.ndim != 2:
			raise ValueError("X must be a 2D array.")
		if X.shape[0] != y.shape[0]:
			raise ValueError("X and y must have the same number of samples.")

		self._validate_params()  # Input guards for the simplified implementation.

		self.X_train = X
		self.y_train = y
		self.classes_ = np.unique(y)
		# Map raw labels to indices so we can keep bincount logic simple.
		self._class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
		return self

	def _validate_params(self):
		if self.weights not in ("uniform", "distance"):
			raise ValueError("Only weights='uniform' or 'distance' are supported for simplicity.")

		unsupported_algos = ("auto", "ball_tree", "kd_tree")
		if self.algorithm in unsupported_algos:
			raise ValueError(f"Algorithm '{self.algorithm}' is not supported in this simplified version.")
		if self.algorithm != "brute":
			raise ValueError("Only algorithm='brute' is supported for simplicity.")

		if callable(self.metric):
			raise ValueError("Callable metrics are not supported for simplicity.")
		if isinstance(self.metric, str):
			if self.metric == "precomputed":
				raise ValueError("Precomputed distance matrices are not supported for simplicity.")
			if self.metric != "minkowski":
				raise ValueError("Only metric='minkowski' is supported without extra dependencies.")
		else:
			raise ValueError("Metric must be 'minkowski' in this simplified version.")

		if self.p <= 0:
			raise ValueError("Parameter p must be greater than 0 for the Minkowski metric.")

	def _compute_distances(self, x):
		# Minkowski distance with order p (same as Euclidean when p=2).
		diff = np.abs(self.X_train - x)
		return np.linalg.norm(diff, ord=self.p, axis=1)

	def _neighbors_within_radius(self, x):
		distances = self._compute_distances(x)
		in_radius = distances <= self.radius
		return distances[in_radius], self.y_train[in_radius]

	def _count_by_class(self, labels, weights):
		counts = np.zeros(len(self.classes_), dtype=float)
		for label, weight in zip(labels, weights):
			counts[self._class_to_index[label]] += weight
		return counts

	def _resolve_empty_neighbors(self):
		if self.outlier_label is None:
			raise ValueError("No neighbors found within the given radius.")
		return self.outlier_label

	def _compute_weights(self, distances):
		if self.weights == "uniform":
			return np.ones_like(distances, dtype=float)

		# Distance weights: closer points get higher weight.
		zero_mask = distances == 0
		if zero_mask.any():
			# Match scikit-learn behavior of relying on exact matches first.
			return np.where(zero_mask, 1.0, 0.0)
		return 1.0 / distances

	def predict(self, X):
		X = np.asarray(X)
		if X.ndim == 1:
			X = X.reshape(1, -1)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array.")
		predictions = []

		for x in X:
			neighbor_distances, neighbor_labels = self._neighbors_within_radius(x)
			if len(neighbor_labels) == 0:
				predictions.append(self._resolve_empty_neighbors())
				continue

			weights = self._compute_weights(neighbor_distances)
			counts = self._count_by_class(neighbor_labels, weights)
			predictions.append(self.classes_[np.argmax(counts)])

		return np.asarray(predictions)

	def predict_proba(self, X):
		X = np.asarray(X)
		if X.ndim == 1:
			X = X.reshape(1, -1)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array.")
		probabilities = []

		for x in X:
			neighbor_distances, neighbor_labels = self._neighbors_within_radius(x)
			if len(neighbor_labels) == 0:
				if self.outlier_label is None:
					raise ValueError("No neighbors found within the given radius.")
				# When we fall back to outlier_label, return zero probability for training classes.
				probabilities.append(np.zeros(len(self.classes_)))
				continue

			weights = self._compute_weights(neighbor_distances)
			counts = self._count_by_class(neighbor_labels, weights)
			probabilities.append(counts / counts.sum())

		return np.vstack(probabilities)
