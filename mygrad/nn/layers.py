import numpy as np
import numpy.typing as npt
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


class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = Value(np.random.randn(vocab_size, embed_dim) * 0.1)
    
    def __call__(self, indices: npt.NDArray):
        indices = np.array(indices, dtype=int)
        out = Value(self.W.data[indices], (self.W,), 'embedding')

        def _backward():
            grad_W = np.zeros_like(self.W.data)
            np.add.at(grad_W, indices, out.grad)
            self.W.grad += grad_W
        
        out._backward = _backward
        return out

    def parameters(self):
        return [self.W]


def add_bias(x: Value, b: Value):
    out = Value(x.data + b.data, (x, b), '+')

    def _backward():
        x.grad += out.grad
        # Sum gradients across the batch dimension before adding to bias grad
        b.grad += out.grad.sum(axis=0, keepdims=True)

    out._backward = _backward
    return out