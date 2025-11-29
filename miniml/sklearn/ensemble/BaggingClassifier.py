import numpy as np
import warnings
import copy
from miniml.sklearn.base import BaseEstimator


class BaggingClassifier(BaseEstimator):
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    # -------------------------------
    #  validate_data()
    # -------------------------------
    def _validate_data(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}.")
        return X, y

    # -------------------------------
    #  _check_sample_weight()
    # -------------------------------
    def _check_sample_weight(self, sample_weight, X):
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.ndim != 1:
            raise ValueError("sample_weight must be 1D array")
        if len(sample_weight) != X.shape[0]:
            raise ValueError("sample_weight length must match X samples")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must be non-negative")
        return sample_weight

    # -------------------------------
    #  _raise_for_params()
    # -------------------------------
    def _raise_for_params(self, fit_params):
        if not isinstance(fit_params, dict):
            raise TypeError("fit_params must be a dictionary")
        # check parameter is dict.
        return True

    # -------------------------------
    # fit() 
    # -------------------------------
    def fit(self, X, y, sample_weight=None, **fit_params):
        # Step 1: check data
        X, y = self._validate_data(X, y)

        # Step 2: check fit_params
        self._raise_for_params(fit_params)

        # Step 3: sample_weight
        if sample_weight is not None:
            sample_weight = self._check_sample_weight(sample_weight, X)
            if not self.bootstrap:
                warnings.warn(
                    f"When fitting {self.__class__.__name__} with sample_weight, "
                    f"it is recommended to use bootstrap=True, got {self.bootstrap}.",
                    UserWarning,
                )

        # Step 4: _fit training
        return self._fit(
            X,
            y,
            max_samples=self.max_samples,
            sample_weight=sample_weight,
            **fit_params,
        )

    # -------------------------------
    # _fit()
    # -------------------------------
    def _fit(self, X, y, max_samples=1.0, sample_weight=None, **fit_params):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        n_sub = int(max_samples * n_samples) if max_samples <= 1 else int(max_samples)

        self.classes_ = np.unique(y)

        self.estimators_ = []
        self.indices_ = []

        for i in range(self.n_estimators):
            # Sample index
            if self.bootstrap:
                indices = rng.integers(0, n_samples, n_sub)
            else:
                indices = rng.choice(n_samples, n_sub, replace=False)

            X_subset = X[indices]
            y_subset = y[indices]

            # Check sample_weight
            if sample_weight is not None:
                sample_weight_subset = sample_weight[indices]
            else:
                sample_weight_subset = None

            # Copy base estimator
            if self.estimator is None:
                raise ValueError("You must provide an estimator to BaggingClassifier.")
            est = copy.deepcopy(self.estimator)

            # Call fit()
            if sample_weight_subset is not None:
                try:
                    est.fit(X_subset, y_subset, sample_weight=sample_weight_subset, **fit_params)
                except TypeError:
                    est.fit(X_subset, y_subset, **fit_params)
            else:
                est.fit(X_subset, y_subset, **fit_params)

            # Save model and index
            self.estimators_.append(est)
            self.indices_.append(indices)

            if self.verbose > 0:
                print(f"Trained estimator {i+1}/{self.n_estimators}")

        return self
    
    def predict(self, X, **params):
        predicted_probability = self.predict_proba(X, **params)
        indices = np.argmax(predicted_probability, axis=1)
        y_pred = self.classes_[indices]
        return y_pred



    def predict_proba(self, X, **params):
        if not hasattr(self, "estimators_"):
            raise ValueError(
                "You should call fit() before predict()"
            )

        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        total_proba = np.zeros((n_samples, n_classes))

        has_proba = hasattr(self.estimators_[0], "predict_proba")

        for est in self.estimators_:
            if has_proba:
                proba_subset = est.predict_proba(X, **params)

                if n_classes == len(est.classes_):
                    if np.all(self.classes_ == est.classes_):
                        total_proba += proba_subset
                    else:
                        indices = np.searchsorted(est.classes_, self.classes_)
                        total_proba += proba_subset[:, indices]
                else:
                    for i, class_label in enumerate(self.classes_):
                        try:
                            est_class_index = np.where(est.classes_ == class_label)[0][0]
                            total_proba[:, i] += proba_subset[:, est_class_index]
                        except IndexError:
                            pass 
            
            else:
                predictions = est.predict(X, **params)
                proba_subset = np.equal.outer(predictions, self.classes_).astype(float)
                total_proba += proba_subset


        avg_proba = total_proba / len(self.estimators_)

        return avg_proba        


    