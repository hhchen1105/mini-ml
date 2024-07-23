# mini-ml

## About

This project aims to provide a backbone implementation of the key machine learning algorithms. We focus on code interpretability over efficiency.

## Design principle

1. Simplicity favours regularity
   
  - We will follow the conventions of scikit-learn's APIs whenever possible. Below are examples of using the linear regression model in scikit-learn and mini-ml, respectively.

    ```python
    # scikit-learn
    from sklearn.linear_model import LinearRegression
    import numpy as np
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    ```

    ```python
    # mini-ml
    from miniml.sklearn.linear_model import LinearRegression
    import numpy as np
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    ```

2. Good design demands reasonable compromises

  - We remove infrequently used parameters in the methods. For example, scikit-learn's linear regression class (`sklearn.linear_model.LinearRegression`) includes parameters `copy_X` (bool), `n_jobs` (int), and `positive` (bool), representing whether copying input feature or not, the number of jobs to use for the computation, and whether forcing the coefficients to be positive, respectively. We ignore these parameters in our implementation.
    

## Similar projects
- [ML from scratch](https://github.com/eriklindernoren/ML-From-Scratch) - ``ML from scratch'' has excellent implementation on many ML models. However, the project has been inactive since 2019.

## TODOs
- Add an installation guide
- Add fundamental supervised learning algorithms: linear regression, logistic regression, decision tree, random forest, SVM, and KNN
