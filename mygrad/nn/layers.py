import numpy as np
from mygrad.core.value import Value


class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = Value(np.random.randn(in_dim, out_dim) * 0.1)
        self.b = Value(np.zeros((1, out_dim)))
    
    def __call__(self, x: Value):
        out = x @ self.W
        return add_bias(out, self.b)
    
    def parameters(self):
        return [self.W, self.b]


def add_bias(x: Value, b: Value):
    out = Value(x.data + b.data, (x, b), '+')

    def _backward():
        x.grad += out.grad
        # Sum gradients across the batch dimension before adding to bias grad
        b.grad += out.grad.sum(axis=0, keepdims=True)

    out._backward = _backward
    return out