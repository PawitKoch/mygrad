import numpy as np


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
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __add__(self, other):
        '''Addition op'''
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

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
            self.grad += np.tranpose(out.grad, inv_axes)
        
        out._backward = _backward
        return out

    @property
    def T(self):
        '''Transpose the value'''
        return self.transpose()
    
    

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