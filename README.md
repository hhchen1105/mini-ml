# mini-ml

## About

This project aims to provide a backbone implementation of the key machine learning algorithms for educational purposes.

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

2. Readability over efficiency

    - This project is mainly for educational purposes. Thus, we value code readability over code efficiency.

3. Good design demands reasonable compromises

    - While our APIs primarily follow scikit-learn's design, we have removed the infrequently used parameters from the corresponding scikit-learn APIs. For example, scikit-learn's linear regression class (`sklearn.linear_model.LinearRegression`) includes parameters `copy_X` (bool), `n_jobs` (int), and `positive` (bool), representing whether copying input feature or not, the number of jobs to use for the computation, and whether forcing the coefficients to be positive, respectively. We ignore these parameters in our implementation.

## Supported algorithms

1. Supervised: k-nearest neighbors, decision tree regressor, decision tree classifier, linear regression, sgd regression.

2. Unsupervised: k-means.

## Installation
1. Change the directory to the folder that contains the mini-ml folder.

   ```bash
   cd [DownloadFolder]
   ```

2. Install the package by running
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
