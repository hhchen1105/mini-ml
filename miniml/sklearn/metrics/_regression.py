import numpy as np

def _check_regression_targets(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape}, y_pred shape {y_pred.shape}")
    
    if not np.issubdtype(y_true.dtype, np.number) or not np.issubdtype(y_pred.dtype, np.number):
        raise ValueError("Both y_true and y_pred must be numeric.")
    
    return True

def mean_squared_error(y_true, y_pred):
    _check_regression_targets(y_true=y_true,y_pred=y_pred)
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    _check_regression_targets(y_true=y_true,y_pred=y_pred)
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    _check_regression_targets(y_true=y_true,y_pred=y_pred)
    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
    residual_variance = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_variance / total_variance)