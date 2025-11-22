import numpy as np
from typing import Any


class BernoulliNB:
    def __init__(self, alpha:float=1.0, force_alpha:bool=True, binarize:float=0.0, fit_prior:bool=True, class_prior:Any|None=None):
        self.alpha = alpha                      # p = p+alpha
        self.force_alpha = force_alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.class_count_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)                                    # class labels (n_classes,)
        self.class_count_ = np.zeros_like(self.classes_)                # each classes' count (n_classes,)
        self.class_log_prior_ = np.array([])                            # each classes' log prob. (n_classes,)
        self.feature_count_ = np.zeros((len(self.classes_),len(X[0])))   # feature counts for each classes
        sample_sum = len(X)
        
        bin_X = self.binarizer(X)

        # read inputs
        for x,yy in zip(bin_X,y):
            same_classes = np.where(self.classes_==yy)[0]
            
            class_idx = same_classes[0]
            # deal with class
            self.class_count_[ class_idx ] += 1
            
            # deal with feature
            for i,xx in enumerate(x):
                self.feature_count_[class_idx][i] += xx
        
        if self.class_prior:
            # calc class log
            self.class_log_prior_ = np.array([ np.log(prior_p) for prior_p in self.class_prior])
        elif not self.fit_prior:
            self.class_log_prior_ = np.array([ np.log(1/len(self.classes_)) for _ in range(len(self.classes_))])
        else:
            self.class_log_prior_ = np.array([ np.log(count/sample_sum) for count in self.class_count_])
    
    def predict(self, X):
        result = np.zeros(len(X))
        cur_prob = 0
        max_prob = 0
        bin_X = self.binarizer(X)

        for i,x in enumerate(bin_X):
            assert(len(x) == len(self.feature_count_[0]))

            for class_idx, counts in enumerate(self.feature_count_):
                cur_prob = self.class_log_prior_[class_idx]
                class_count_i = self.class_count_[class_idx]
                for xx, count in zip(x, counts):
                    cur_prob += np.log(xx*(count+self.alpha)/(class_count_i + 2*self.alpha) + \
                                       (1-xx)*(1-(count+self.alpha)/(class_count_i + 2*self.alpha))) 
                
                if class_idx==0 or cur_prob > max_prob:
                    max_prob = cur_prob
                    result[i] = self.classes_[class_idx]

        return result

    def predict_proba(self, X):
        result = np.empty((len(X),len(self.classes_)))
        cur_prob = 0
        bin_X = self.binarizer(X)

        for i,x in enumerate(bin_X):
            assert(len(x) == len(self.feature_count_[0]))

            for class_idx, counts in enumerate(self.feature_count_):
                cur_prob = self.class_log_prior_[class_idx]
                class_count_i = self.class_count_[class_idx]
                for xx, count in zip(x, counts):
                    cur_prob += np.log(xx*(count+self.alpha)/(class_count_i + 2*self.alpha) + \
                                       (1-xx)*(1-(count+self.alpha)/(class_count_i + 2*self.alpha))) 
                
                result[i][class_idx] = np.exp(cur_prob)

            prob_sum = np.sum(result[i])
            result[i] = result[i]/prob_sum
        
        return result
    
    def binarizer(self, X):
        bin_X = np.empty(X.shape)
        bin_X[X <= self.binarize] = 0
        bin_X[X >  self.binarize] = 1
        return bin_X