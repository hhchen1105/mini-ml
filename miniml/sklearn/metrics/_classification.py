import numpy as np

def _check_classification_targets(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape}, y_pred shape {y_pred.shape}")
    
    if not np.issubdtype(y_true.dtype, np.integer) or not np.issubdtype(y_pred.dtype, np.integer):
        raise ValueError("Both y_true and y_pred must be integers representing class labels.")
    
    return True

def accuracy_score(y_true, y_pred):
    _check_classification_targets(y_true=y_true,y_pred=y_pred)
    return np.sum(y_true == y_pred) / len(y_true)
