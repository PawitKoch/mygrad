import numpy as np
from .utils import unbroadcast


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data: float | np.ndarray = data
        self.grad: float | np.ndarray = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(c for c in _children if isinstance(c, Value))
        self._op = _op
    
    def __neg__(self):
        '''Negation op'''
        out = Value(-self.data, (self,), 'neg')

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        '''Multiplication op'''
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out
    
    def __add__(self, other):
        '''Addition op'''
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        '''Subtraction op'''
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    
    def __truediv__(self, other):
        '''Division op'''
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += (1.0 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        '''Power op'''
        assert isinstance(other, (int, float)), "Power must be an integer or float"
        out = Value(self.data ** other, (self,), f'pow({other})')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        '''Matrix multiplication op'''
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        '''Get item at index'''
        out = Value(self.data[idx], (self,), 'getitem')

        def _backward():
            grad = np.zeros_like(self.data)
            np.add.at(grad, idx, out.grad)
            self.grad += grad

        out._backward = _backward
        return out

    def reshape(self, new_shape):
        '''Reshape value to new shape'''
        out = Value(self.data.reshape(new_shape), (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        
        out._backward = _backward
        return out

    def transpose(self, axes=None):
        '''Transpose along specified axes'''
        out = Value(np.transpose(self.data, axes), (self,), 'transpose')
        if axes is None:
            inv_axes = None
        else:
            inv_axes = np.argsort(axes)
        
        def _backward():
            self.grad += np.transpose(out.grad, inv_axes)
        
        out._backward = _backward
        return out

    @property
    def T(self):
        '''Transpose the value'''
        return self.transpose()
    
    def sum(self, axis=None, keepdims=False):
        '''Sum the value along the specified axis.'''
        out = Value(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        if isinstance(axis, int):
            axis = (axis,)

        def _backward():
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
        '''Mean the value along the specified axis.'''
        out = Value(self.data.mean(axis=axis, keepdims=keepdims), (self,), 'mean')
        if isinstance(axis, int):
            axis = (axis,)
        if axis is None:
            N = self.data.size
        else:
            N = 1
            for ax in axis:
                N *= self.data.shape[ax]            

        def _backward():
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
        '''Adds a dimension of size 1 along specified axis.'''
        out = Value(np.expand_dims(self.data, axis), (self,), 'expand_dims')
        def _backward():
            g = out.grad
            g += np.squeeze(g, axis=axis)
            self.grad += g

        out._backward = _backward
        return out

    def squeeze(self, axis=None):
        '''Remove dimensions of size 1 along specified axes.'''
        out = Value(np.squeeze(self.data, axis=axis), (self,), 'squeeze')
        if axis is None:
            removed = tuple(i for i, s in enumerate(self.data.shape) if s == 1)
        else:
            if isinstance(axis, int):
                removed = (axis,)
            else:
                removed = tuple(axis)
        
        def _backward():
            g = out.grad
            for ax in sorted(removed):
                g = np.expand_dims(g, ax)
            self.grad += g
        
        out._backward = _backward
        return out

    def backward(self):
        if isinstance(self.data, np.ndarray):
            self.grad = np.ones_like(self.data)
        else:
            self.grad = 1.0

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        for node in reversed(topo):
            node._backward()