from math import pi, cos


class WarmUpCosineAnnealing:
    def __init__(self, lr_min: float, lr_max: float,  warm_up: int = 10):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up

    def compute_lr(self, current_iter: int, max_iter: int, current_lr: float) -> float:
        if self.lr_min == self.lr_max: 
            return self.lr_max
        
        if current_iter < self.warm_up:
            # Warm up
            current_lr = self.lr_max / self.warm_up * (current_iter + 1)
        else:
            # Cosine annealing
            max_iter -= self.warm_up
            current_iter -= self.warm_up
            current_lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1 + cos(current_iter * pi / max_iter))

        return current_lr
