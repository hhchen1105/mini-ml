# mini-ml

## About

This project aims to provide a backbone implementation of the key machine learning algorithms. We focus on code interpretability over efficiency.

## API design principle

We will follow the APIs used in scikit-learn whenever possible. Below are examples of using the linear regression model in scikit-learn and mini-ml, respectively.

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

## Similar projects
- [ML from scratch](https://github.com/eriklindernoren/ML-From-Scratch) - ``ML from scratch'' has excellent implementation on many ML models. However, the project has been inactive since 2019.

## TODOs
- Add an installation guide
- Add fundamental supervised learning algorithms: linear regression, logistic regression, decision tree, random forest, SVM, and KNN
