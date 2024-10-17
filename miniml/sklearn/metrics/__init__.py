from ._classification import(
    accuracy_score
)

from ._regression import(
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

__all__ = ['accuracy_score',"mean_squared_error", "mean_absolute_error", "r2_score"]