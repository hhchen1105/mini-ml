from .Learning_Rate_Scheduler import WarmUpCosineAnnealing
from .Solver import AdamWSolver
from .Activation_Function import SigmoidActivation, LeakyReLUActivation, IdentityActivation, SoftmaxActivation
from .Loss_Function import MSE, CrossEntropy_MultiClass


class Activations:
    ACTIVATION_FUNCTION_MAP = {"sigmoid": SigmoidActivation,
                               "leaky_relu": LeakyReLUActivation,
                               "identity": IdentityActivation,
                               "softmax": SoftmaxActivation}

    @classmethod
    def get_activation(cls, activation: str, **kwargs) -> object:
        return cls.ACTIVATION_FUNCTION_MAP[activation](**kwargs)


class LearningRates:
    LEARNING_RATE_MAP = {"warmup_cosine_annealing": WarmUpCosineAnnealing}

    @classmethod
    def get_learning_rate(cls, learning_rate: str, **kwargs) -> object:
        return cls.LEARNING_RATE_MAP[learning_rate](**kwargs)


class Solvers:
    SOLVER_MAP = {"adamw": AdamWSolver}

    @classmethod
    def get_solver(cls, solver: str, **kwargs) -> object:
        return cls.SOLVER_MAP[solver](**kwargs)


class Losses:
    LOSS_MAP = {"mse": MSE,
                "cross_entropy_multiclass": CrossEntropy_MultiClass}

    @classmethod
    def get_loss(cls, loss: str, **kwargs) -> object:
        return cls.LOSS_MAP[loss](**kwargs)
