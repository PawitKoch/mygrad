import numpy as np
from mygrad.core.value import Value


class SGD:
    def __init__(self, parameters: list[Value], lr=0.1):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
