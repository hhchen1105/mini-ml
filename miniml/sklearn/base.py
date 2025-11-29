# miniml\sklearn\base.py

import inspect


class BaseEstimator:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def _get_param_names(cls):
        """Get hyper_parameter names for the estimator"""
        # get int class __init__
        init = cls.__init__

        # deal with no explicit __init__
        if init is object.__init__:
            return []

        # using inspect to get the function signature
        init_signature = inspect.signature(init)
        # filter out 'self'
        parameters = [
            p.name for p in init_signature.parameters.values() if p.name != "self"
        ]
        # return sorted parameters
        return sorted(parameters)

    def get_params(self):
        """Get hyper_parameters of the estimator"""
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            out[key] = value
        return out
    
    def set_params(self, **params):
        """
        Set the hyper_parameters of the estimator.
        """
        if not params:
            return self

        # Get the dictionary of all valid parameters of the current model
        valid_params = self.get_params()

        for key, value in params.items():
            # Validate if the parameter exists (keep this check to help users avoid errors)
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {list(valid_params.keys())!r}."
                )

            # Use setattr to set the attribute
            setattr(self, key, value)

        return self

    def save_model(self, filepath):
        import pickle

        with open(
            filepath, "wb"
        ) as f:  # save the entire instance (self), which includes all trained parameters
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath):
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)
