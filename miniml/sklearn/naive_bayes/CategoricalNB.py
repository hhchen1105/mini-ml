import numpy as np

class CategoricalNB:
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, min_categories=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.min_categories = min_categories

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # 1. Metadata extraction
        self.n_samples, self.n_features_in_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # 2. Determine n_categories per feature
        if self.min_categories is None:
            self.n_categories_ = [np.max(X[:, i]) + 1 for i in range(self.n_features_in_)]
        elif isinstance(self.min_categories, int):
            self.n_categories_ = [max(self.min_categories, np.max(X[:, i]) + 1) 
                                  for i in range(self.n_features_in_)]
        else:
            self.n_categories_ = self.min_categories
            
        # 3. Calculate the number of each class 
        self.class_count_ = np.zeros(self.n_classes_)
        for i, c in enumerate(self.classes_):
            self.class_count_[i] = np.sum(y == c)
            
        # 4. Calculate Class Log Priors
        if self.class_prior is not None:
            self.class_log_prior_ = np.log(self.class_prior)
        elif self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_) - np.log(self.n_samples)
        else:
            self.class_log_prior_ = np.zeros(self.n_classes_) - np.log(self.n_classes_)
            
        # 5. Calculate Feature Counts and Log Probabilities
        self.category_count_ = []
        self.feature_log_prob_ = []
        
        for i in range(self.n_features_in_):
            n_cat = int(self.n_categories_[i])
            # shape: (n_classes, n_categories_for_feature_i)
            count_matrix = np.zeros((self.n_classes_, n_cat))
            
            for c_idx, c in enumerate(self.classes_):
                X_c = X[y == c, i]
                # Count occurrences of each category in this class
                counts = np.bincount(X_c, minlength=n_cat)
                if len(counts) > n_cat:
                     counts = counts[:n_cat]
                count_matrix[c_idx, :] = counts
            
            self.category_count_.append(count_matrix)
            
            # Smoothed Probability Calculation
            # P(x_i=t | y=c) = (N_tic + alpha) / (N_c + alpha * n_categories_i)
            numerator = count_matrix + self.alpha
            # Denominator shape must broadcast: (n_classes, 1)
            denominator = (self.class_count_ + self.alpha * n_cat).reshape(-1, 1)
            
            feature_log_prob = np.log(numerator) - np.log(denominator)
            self.feature_log_prob_.append(feature_log_prob)
            
        return self

    def predict_log_proba(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Start with class priors
        log_proba = np.zeros((n_samples, self.n_classes_))
        log_proba += self.class_log_prior_
        
        # Add log-likelihoods for each feature
        for i in range(self.n_features_in_):
            # feature_log_prob_[i] shape: (n_classes, n_categories)
            # X[:, i] values are indices into the columns of feature_log_prob_
            term = self.feature_log_prob_[i][:, X[:, i]].T
            log_proba += term
            
        return log_proba

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        # Return the class with the highest log probability
        return self.classes_[np.argmax(log_proba, axis=1)]
    
    def predict_proba(self, X):
        log_proba = self.predict_log_proba(X)
        # Use log-sum-exp trick for numerical stability
        max_log = np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba - max_log)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba
        
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
