import numpy as np
import numpy.typing as npt
from mygrad.core.value import Value
from mygrad.core.ops import softmax


class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = Value(np.random.randn(in_dim, out_dim) * 0.1)
        self.b = Value(np.zeros((1, out_dim)))
    
    def __call__(self, x: Value):
        out = x @ self.W
        return out + self.b
    
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


class SelfAttention:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.scale = np.sqrt(embed_dim)

        # projections for Q, K, V
        self.W_q = Linear(embed_dim, embed_dim)
        self.W_k = Linear(embed_dim, embed_dim)
        self.W_v = Linear(embed_dim, embed_dim)
    
    def __call__(self, x: Value):
        """
        x: (batch_size, seq_length, embed_dim)
        returns: (batch_size, seq_length, embed_dim)
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = (Q @ K.transpose((0, 2, 1))) / self.scale
        weights = softmax(scores)

        out = weights @ V
        return out

    def parameters(self):
        return self.W_q.parameters() + self.W_k.parameters() + self.W_v.parameters()