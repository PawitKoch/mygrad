from mygrad.core.value import Value
from mygrad.core.ops import relu, sigmoid
from .layers import Linear


class MLP:
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.l1 = Linear(in_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, out_dim)
    
    def __call__(self, x: Value):
        h = relu(self.l1(x))
        out = sigmoid(self.l2(h))
        return out

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()