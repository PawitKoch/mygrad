import numpy as np
from mygrad.core.utils import unbroadcast


class Value:
    """
    Core autograd engine

    The computational graph is a DAG (Directed Acyclic Graph) where:
    - Nodes are Value objects (contain data and gradients)
    - Edges represent operations that produced this Value from parent Values

    Example graph for: d = (a + b) * c

        a (2)      b (3)
         \         /
          \       /
           \     /
            \   /
             (+)  ------>  e (5)       c (4)
                            \           /
                             \         /
                              \       /
                               \     /
                                \   /
                                 (*)  ------>  d (20)

    Backward pass (gradient flow):
    1. Start at output node d, set d.grad = 1.0
    2. Traverse graph in REVERSE topological order
    3. Each node computes gradients for its children using chain rule:
       - d._backward() computes e.grad and c.grad
       - e._backward() computes a.grad and b.grad

    Attributes:
        data: The actual numerical value (scalar or ndarray)
        grad: Accumulated gradient (same shape as data)
        _backward: Function that computes gradients for parent nodes
        _prev: Set of parent Value nodes that created this Value
        _op: String describing the operation (for debugging)
    """

    def __init__(self, data, _children=(), _op=""):
        self.data: float | np.ndarray = data
        self.grad: float | np.ndarray = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(c for c in _children if isinstance(c, Value))
        self._op = _op

    def __neg__(self):
        """Negation op"""
        out = Value(-self.data, (self,), "neg")

        def _backward():
            # Chain rule: if out = -self, then ∂L/∂self = ∂L/∂out * ∂out/∂self
            # ∂out/∂self = ∂(-self)/∂self = -1
            # Therefore: ∂L/∂self = ∂L/∂out * (-1) = -∂L/∂out
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Multiplication op"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # Chain rule: if out = self * other, then:
            # ∂L/∂self = ∂L/∂out * ∂out/∂self = ∂L/∂out * other
            # ∂L/∂other = ∂L/∂out * ∂out/∂other = ∂L/∂out * self
            # Unbroadcast handles when shapes were expanded during forward pass
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __add__(self, other):
        """Addition op"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # Chain rule: if out = self + other, then:
            # ∂L/∂self = ∂L/∂out * ∂out/∂self = ∂L/∂out * 1 = ∂L/∂out
            # ∂L/∂other = ∂L/∂out * ∂out/∂other = ∂L/∂out * 1 = ∂L/∂out
            # Gradients simply pass through unchanged
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        """Subtraction op"""
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __truediv__(self, other):
        """Division op"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")

        def _backward():
            # Chain rule: if out = self / other, then:
            # ∂out/∂self = 1/other
            # ∂L/∂self = ∂L/∂out * (1/other)
            self_grad = (1.0 / other.data) * out.grad
            self.grad += unbroadcast(self_grad, self.data.shape)

            # ∂out/∂other = -self/other² (quotient rule)
            # ∂L/∂other = ∂L/∂out * (-self/other²)
            other_grad = (-self.data / (other.data**2)) * out.grad
            other.grad += unbroadcast(other_grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, other):
        """Power op"""
        assert isinstance(other, (int, float)), "Power must be an integer or float"
        out = Value(self.data**other, (self,), f"pow({other})")

        def _backward():
            # Chain rule: if out = self^n (where n is constant), then:
            # ∂out/∂self = n * self^(n-1) (power rule)
            # ∂L/∂self = ∂L/∂out * n * self^(n-1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Matrix multiplication op"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), "@")

        def _backward():
            # Chain rule for matrix multiplication: if out = self @ other, then:
            # ∂L/∂self = ∂L/∂out @ other^T
            # ∂L/∂other = self^T @ ∂L/∂out
            # Using swapaxes for transpose to handle batched operations (..., i, j) -> (..., j, i)
            self_grad = out.grad @ np.swapaxes(other.data, -1, -2)
            self.grad += unbroadcast(self_grad, self.data.shape)
            other_grad = np.swapaxes(self.data, -1, -2) @ out.grad
            other.grad += unbroadcast(other_grad, other.data.shape)

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        """Get item at index"""
        out = Value(self.data[idx], (self,), "getitem")

        def _backward():
            # Chain rule: if out = self[idx], gradients only flow to indexed positions
            # ∂L/∂self[i] = ∂L/∂out if i in idx, else 0
            # np.add.at handles accumulation for duplicate indices
            grad = np.zeros_like(self.data)
            np.add.at(grad, idx, out.grad)
            self.grad += grad

        out._backward = _backward
        return out

    def reshape(self, new_shape):
        """Reshape value to new shape"""
        out = Value(self.data.reshape(new_shape), (self,), "reshape")

        def _backward():
            # Chain rule: reshape doesn't change values, only their arrangement
            # ∂L/∂self = reshape(∂L/∂out, original_shape)
            # Gradients just need to be reshaped back to match input shape
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, axes: tuple[int] | list[int] | None = None):
        """Transpose along specified axes"""
        out = Value(np.transpose(self.data, axes), (self,), "transpose")
        if axes is None:
            inv_axes = None
        else:
            axes = list(axes) if isinstance(axes, tuple) else axes
            inv_axes = np.argsort(axes)

        def _backward():
            # Chain rule: transpose doesn't change values, only axis ordering
            # If out = transpose(self, axes), then:
            # ∂L/∂self = transpose(∂L/∂out, inverse_axes)
            # The inverse transpose undoes the forward transpose
            self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        return out

    @property
    def T(self):
        """Transpose the value"""
        return self.transpose()

    def sum(self, axis=None, keepdims=False):
        """Sum the value along the specified axis."""
        out = Value(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        if isinstance(axis, int):
            axis = (axis,)

        def _backward():
            # Chain rule: if out = sum(self), then:
            # ∂out/∂self[i] = 1 for all i (every element contributes equally)
            # ∂L/∂self = ∂L/∂out * 1 = ∂L/∂out broadcasted to all positions
            # The gradient is broadcast back to match the original shape
            g = out.grad
            if axis is None:
                g = np.broadcast_to(g, self.data.shape)
            else:
                if not keepdims:
                    for ax in sorted(axis):
                        g = np.expand_dims(g, ax)
                g = np.broadcast_to(g, self.data.shape)
            self.grad += g

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean the value along the specified axis."""
        out = Value(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")
        if isinstance(axis, int):
            axis = (axis,)
        if axis is None:
            N = self.data.size
        else:
            N = 1
            for ax in axis:
                N *= self.data.shape[ax]

        def _backward():
            # Chain rule: if out = mean(self) = sum(self)/N, then:
            # ∂out/∂self[i] = 1/N for all i
            # ∂L/∂self = ∂L/∂out * (1/N)
            # Divide by N and broadcast back to original shape
            g = out.grad / N
            if axis is None:
                g = np.broadcast_to(g, self.data.shape)
            else:
                if not keepdims:
                    for ax in sorted(axis):
                        g = np.expand_dims(g, ax)
                g = np.broadcast_to(g, self.data.shape)
            self.grad += g

        out._backward = _backward
        return out

    def expand_dims(self, axis):
        """Adds a dimension of size 1 along specified axis."""
        out = Value(np.expand_dims(self.data, axis), (self,), "expand_dims")

        def _backward():
            # Chain rule: expand_dims adds singleton dimensions
            # ∂L/∂self = squeeze(∂L/∂out, axis) to remove the added dimension
            # Gradients flow through unchanged, just need shape adjustment
            g = out.grad
            g += np.squeeze(g, axis=axis)
            self.grad += g

        out._backward = _backward
        return out

    def squeeze(self, axis=None):
        """Remove dimensions of size 1 along specified axes."""
        out = Value(np.squeeze(self.data, axis=axis), (self,), "squeeze")
        if axis is None:
            removed = tuple(i for i, s in enumerate(self.data.shape) if s == 1)
        else:
            if isinstance(axis, int):
                removed = (axis,)
            else:
                removed = tuple(axis)

        def _backward():
            # Chain rule: squeeze removes singleton dimensions
            # ∂L/∂self = expand_dims(∂L/∂out, removed_axes) to restore shape
            # Gradients flow through unchanged, just need shape adjustment
            g = out.grad
            for ax in sorted(removed):
                g = np.expand_dims(g, ax)
            self.grad += g

        out._backward = _backward
        return out

    def backward(self):
        """
        Compute gradients via backpropagation using reverse-mode autodiff.

        Algorithm:
        1. Set this node's gradient to 1 (since ∂L/∂L = 1)
        2. Build topological ordering of all nodes in computation graph
        3. Traverse in REVERSE order, calling each node's _backward()

        Topological sort ensures parents are processed before children,
        so gradients accumulate correctly via chain rule.

        Example:
            loss = (x * w + b).sum()  # creates graph: x -> (*) -> (+) -> sum -> loss
            loss.backward()            # flows gradients: loss -> sum -> (+) -> (*) -> x,w,b
        """
        # Initialize gradient: dL/dL = 1
        if isinstance(self.data, np.ndarray):
            self.grad = np.ones_like(self.data)
        else:
            self.grad = 1.0

        # Build topological order: parents before children
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backpropagate: reverse order so gradients flow correctly
        for node in reversed(topo):
            node._backward()
