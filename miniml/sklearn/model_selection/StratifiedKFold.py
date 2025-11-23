import numpy as np
from sklearn.model_selection import StratifiedKFold

# Example data
X = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40],
    [5, 50],
    [6, 60],
    [7, 70],
    [8, 80],
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # labels (stratified by class)

# Create StratifiedKFold object
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# core method: get_n_splits
print("Number of splits:", skf.get_n_splits(X, y))
# (X, y args are optional; included for clarity)

# core method: split
fold = 1
for train_idx, val_idx in skf.split(X, y):
    print(f"\nFold {fold}")
    print("Train indices:", train_idx)
    print("Val indices:", val_idx)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print("Train class distribution:", np.bincount(y_train))
    print("Val class distribution:", np.bincount(y_val))

    fold += 1
