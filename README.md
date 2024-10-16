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

    | model name               | API                                            |
    |--------------------------|------------------------------------------------|
    | k-nearest neighbors      | miniml.sklearn.neighbors.KNeighborsClassifier  |
    | decision tree classifier | miniml.sklearn.tree.DecisionTreeClassifier     |
    | decision tree regressor  | miniml.sklearn.tree.DecisionTreeRegressor      |
    | random forest classifier | miniml.sklearn.ensemble.RandomForestClassifier |
    | random forest regressor  | miniml.sklearn.ensemble.RandomForestRegressor  |
    | linear regression        | miniml.sklearn.linear_model.LinearRegressor    |
    | SGD regression           | miniml.sklearn.linear_model.SGDRegressor       |
    | Ridge regression         | miniml.sklearn.linear_model.Ridge              |
    |                          |                                                |

1. Unsupervised:

    | model name               | API                                            |
    |--------------------------|------------------------------------------------|
    | k-means                  | miniml.sklearn.cluster.KMeans                  |

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
