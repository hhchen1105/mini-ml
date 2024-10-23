# mini-ml

## About

This project provides a foundational implementation of key machine learning algorithms for educational and research purposes.

From an **educational** perspective, while libraries like Scikit-learn prioritize robustness and efficiency by using complex and optimized code, our implementation focuses on readability above all else. This makes it easier for those looking to understand the inner workings of machine learning algorithms to follow along and grasp the details through the code.

From a **research** standpoint, having a clear and flexible backbone implementation allows for faster experimentation and modification of existing machine learning algorithms. This can significantly streamline the process of improving or testing new algorithm designs.

**In summary**, this project serves as a resource for both learners and researchers, providing clear, accessible code that encourages exploration and experimentation with machine learning algorithms without compromising on essential functionality.

## Design principle

1. Least surprise
   
    - We will follow the conventions of scikit-learn's APIs whenever possible. Below are examples of using the linear regression model in scikit-learn and mini-ml, respectively.

    ```python
    # scikit-learn
    from sklearn.linear_model import LinearRegression
    import numpy as np
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression()
    reg.fit(X, y)
    ```

    ```python
    # mini-ml
    from miniml.sklearn.linear_model import LinearRegression
    import numpy as np
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression()
    reg.fit(X, y)
    ```

1. Readability over efficiency

    - This project is mainly for educational and research purposes. Thus, we value code readability over code efficiency.

1. Good design demands reasonable compromises

    - While our APIs primarily follow scikit-learn's design, we have removed the infrequently used parameters and functions from the corresponding scikit-learn APIs. For example, scikit-learn's linear regression class (`sklearn.linear_model.LinearRegression`) includes parameters `copy_X` (bool), `n_jobs` (int), and `positive` (bool), representing whether copying input feature or not, the number of jobs to use for the computation, and whether forcing the coefficients to be positive, respectively. We ignore these parameters in our implementation.

## Supported algorithms

1. Supervised:

    | model name               | API                                            | key concept
    |--------------------------|------------------------------------------------|-------------
    | k-nearest neighbors      | miniml.sklearn.neighbors.KNeighborsClassifier  | Classifies a data point based on the majority class of its k-nearest neighbors.
    | decision tree classifier | miniml.sklearn.tree.DecisionTreeClassifier     | Builds a decision tree by recursively splitting the data based on feature values to classify data.
    | decision tree regressor  | miniml.sklearn.tree.DecisionTreeRegressor      | Builds a decision tree that predicts continuous values by splitting the data recursively.
    | random forest classifier | miniml.sklearn.ensemble.RandomForestClassifier | An ensemble of decision trees where each tree votes, and the majority vote is the final classification.
    | random forest regressor  | miniml.sklearn.ensemble.RandomForestRegressor  | An ensemble of decision trees used to predict continuous values by averaging predictions from multiple trees.
    | linear regression        | miniml.sklearn.linear_model.LinearRegressor    | Predicts a continuous target value by fitting a linear relationship between input features and the target. It solves for the regression coefficients by using the pseudo-inverse of the feature matrix, computed via Singular Value Decomposition (SVD)
    | SGD regression           | miniml.sklearn.linear_model.SGDRegressor       | Performs linear regression using Stochastic Gradient Descent for optimization.
    | Ridge regression         | miniml.sklearn.linear_model.Ridge              | A variation of linear regression that includes L2 regularization to prevent overfitting. It solves for the regression coefficients using the pseudo-inverse of the regularized feature matrix, computed via Singular Value Decomposition (SVD).
    | Lasso regression         | miniml.sklearn.linear_model.Lasso              | A variation of linear regression that includes L1 regularization to promote sparse feature selection. It is solved using coordinate descent.
    | ElasticNet regression    | miniml.sklearn.linear_model.ElasticNet         | A linear regression model that combines both L1 (Lasso) and L2 (Ridge) regularization to improve prediction accuracy and model interpretability by balancing feature selection (sparsity) and coefficient shrinkage. It is solved using coordinate descent.

1. Unsupervised:

    | model name               | API                                            | key concept
    |--------------------------|------------------------------------------------|----------------
    | k-means                  | miniml.sklearn.cluster.KMeans                  | Partitions the data into k clusters by iteratively assigning data points to the nearest cluster center and updating the cluster centers based on the mean of the points in each cluster.

## Installation
1. Change the directory to the folder that contains the mini-ml folder.

   ```bash
   cd [DownloadFolder]
   ```

1. Install the package by running
   ```bash
   pip install miniml
   ```

## Contribution

We welcome contributions. Here is a short intro on how to contribute.

1. Fork the Repository.

    - Fork the repository to create your own copy. This allows you to freely make changes without affecting the original project.
    
1. Clone Your Fork to your local machine and navigate to the project directory.
   ```bash
   git clone https://github.com/your-username/project-name.git
   cd mini-ml
   ```  

1. Create a New Branch
   ```bash
   git checkout -b feature-branch-name
   ```
     
1. Make Your Changes. Write test cases if you add a new functionality.

1. Run tests.
   ```bash
   pytest
   ```
  
1. Commit and Push Your Changes
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   git push origin feature-branch-name
   ```

1. Submit a Pull Request

## Similar projects
- [ML from scratch](https://github.com/eriklindernoren/ML-From-Scratch) - ``ML from scratch'' has excellent implementation on many ML models. However, the project has been inactive since 2019. Also, the project has its own API design.
