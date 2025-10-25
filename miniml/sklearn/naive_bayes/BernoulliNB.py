import numpy as np
from typing import Any
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


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
                                                                         #    (n_classes, n_features)
        # self.feature_log_prob_  = np.array([])                        # feature log probs for each classes
                                                                        #     (n_classes, n_features)
        # self.n_features_in_
        # self.feature_names_in_
        sample_sum = len(X)
        # feature_sum = np.zeros_like(X[0])
        
        bin_X = self.binarizer(X)

        # read inputs
        for x,yy in zip(bin_X,y):
            same_classes = np.where(self.classes_==yy)[0]
            
            class_idx = same_classes[0]
            # deal with class
            self.class_count_[ class_idx ] += 1
            
            # deal with feature
            for i,xx in enumerate(x):
                # feature_sum[i] += xx
                self.feature_count_[class_idx][i] += xx
        
        if self.class_prior:
            # calc class log
            # print("class_prior specified")
            self.class_log_prior_ = np.array([ np.log(prior_p) for prior_p in self.class_prior])
        elif not self.fit_prior:
            # print("class_prior specified")
            self.class_log_prior_ = np.array([ np.log(1/len(self.classes_)) for _ in range(len(self.classes_))])
        else:
            self.class_log_prior_ = np.array([ np.log(count/sample_sum) for count in self.class_count_])
    
    def predict(self, X):
        result = np.zeros(len(X))
        cur_prob = 0
        max_prob = 0
        # print(f"class_count: {self.class_count_}")
        bin_X = self.binarizer(X)

        for i,x in enumerate(bin_X):
            assert(len(x) == len(self.feature_count_[0]))

            for class_idx, counts in enumerate(self.feature_count_):
                cur_prob = self.class_log_prior_[class_idx]
                class_count_i = self.class_count_[class_idx]
                for xx, count in zip(x, counts):
                    cur_prob += np.log(xx*(count+self.alpha)/(class_count_i + 2*self.alpha) + \
                                       (1-xx)*(1-(count+self.alpha)/(class_count_i + 2*self.alpha))) 
                    # print(f"np.log({self.alpha + xx*count/class_count_i + (1-xx)*(1-count/class_count_i)}) = np.log({xx}*{count}/{class_count_i} + (1-{xx})*(1-{count}/{class_count_i}))")
                
                if class_idx==0 or cur_prob > max_prob:
                    max_prob = cur_prob
                    result[i] = self.classes_[class_idx]

        return result

    def predict_proba(self, X):
        result = np.empty((len(X),len(self.classes_)))
        cur_prob = 0
        # max_prob = 0
        # print(f"class_count: {self.class_count_}")
        bin_X = self.binarizer(X)

        for i,x in enumerate(bin_X):
            assert(len(x) == len(self.feature_count_[0]))

            for class_idx, counts in enumerate(self.feature_count_):
                cur_prob = self.class_log_prior_[class_idx]
                class_count_i = self.class_count_[class_idx]
                for xx, count in zip(x, counts):
                    cur_prob += np.log(xx*(count+self.alpha)/(class_count_i + 2*self.alpha) + \
                                       (1-xx)*(1-(count+self.alpha)/(class_count_i + 2*self.alpha))) 
                    # print(f"np.log({xx*(count+self.alpha)/(class_count_i + 2*self.alpha) + \
                    #                    (1-xx)*(1-(count+self.alpha)/(class_count_i + 2*self.alpha))}... = np.log({xx}*({count}+{self.alpha})/({class_count_i} + 2*{self.alpha})...")
                
                result[i][class_idx] = np.exp(cur_prob)

            prob_sum = np.sum(result[i])
            result[i] = result[i]/prob_sum

            # print(f"self.classes_ = {self.classes_}")
        
        return result
    
    def binarizer(self, X):
        bin_X = np.empty(X.shape)
        bin_X[X <= self.binarize] = 0
        bin_X[X >  self.binarize] = 1
        return bin_X


if __name__ == "__main__":
    b = BernoulliNB(alpha=10, binarize=5, fit_prior=True)
    bn = BernoulliNB(alpha=10, binarize=5, fit_prior=False)
    # b2 = my_BernoulliNB(alpha=10, binarize=5, fit_prior=True)
    # b2n = my_BernoulliNB(alpha=10, binarize=5, fit_prior=False)
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    b.fit(X_train, y_train)
    bn.fit(X_train, y_train)
    # b2.fit(X_train, y_train)
    # b2n.fit(X_train, y_train)

    b_prob = b.predict_proba(X_test)
    bn_prob = bn.predict_proba(X_test)
    # b2_prob = b2.predict_proba(X_test)
    # b2n_prob = b2n.predict_proba(X_test)
    # print(b2_prob == b2n_prob)
    # print(b_prob == bn_prob)
    # print(b_prob == b2_prob)
    # print(bn_prob == b2n_prob)
    # print("compare offical and mine")
    # for bp,b2p in zip(b_prob, b2_prob):
    #     if -0.000001>bp[0]-b2p[0] or bp[0]-b2p[0]>0.000001:
    #         print(bp[0]-b2p[0])
    #     else:
    #         # print("SIMILAR")
    #         pass

    # print("compare offical and mine with no fit prior")
    # for bp,b2p in zip(bn_prob, b2n_prob):
    #     if -0.000001>bp[0]-b2p[0] or bp[0]-b2p[0]>0.000001:
    #         print(bp[0]-b2p[0])
    #     else:
    #         # print("SIMILAR")
    #         pass

    #############################################################  

    # b2.fit(X_train, y_train)

    # classes_ = b2.classes_ 
    # class_count_ = b2.class_count_ 
    # # class_log_prior_ = b2.class_log_prior_ 
    # feature_count_ = b2.feature_count_ 

    # b2.fit2(X_train, y_train)
    # classes_2 = b2.classes_
    # class_count_2 = b2.class_count_
    # # class_log_prior_2 = b2.class_log_prior_
    # feature_count_2 = b2.feature_count_

    # print(feature_count_)
    # print(feature_count_2)

    # print(b_prob == b2_prob)

    # train_data_x = [
    #     # ['x','y','z'],
    #     [0,0,0],
    #     [0,0,1],
    #     [0,1,0],
    #     [0,1,1],
    #     [1,0,0],
    #     [1,0,1],
    #     [1,1,0],
    #     [1,1,1],
    # ]
    # train_data_y = [0.5,1.5,0.5,1.5,0.5,1.5,0.5,1.5]

    # # train_data_x2 = [
    # #     [0,0],
    # #     [0,1],
    # #     [1,0],
    # #     [1,1],
    # # ]
    # # train_data_y2 = [0,0,1,1]

    # b.fit(train_data_x, train_data_y)
    # b2.fit(train_data_x, train_data_y)


    # print(b.predict(train_data_x[-1:]))
    # print(type(b.predict(train_data_x[-1:])[0]))
    # print(b.predict_proba(train_data_x[-1:]))

    # print(b2.predict(train_data_x[-1:]))
    # print(type(b2.predict(train_data_x[-1:])[0]))
    # print(b2.predict_proba(train_data_x[-1:]))
