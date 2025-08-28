import numpy as np
from mygrad.core.value import Value

x = Value(np.arange(6.).reshape(2, 3))
y = x.reshape((3, 2)).sum()
y.backward()
assert np.allclose(x.grad, np.ones_like(x.data))

x = Value(np.random.randn(2, 3))
x.grad[:] = 0 # reset gradient
y = x.T.sum()
y.backward()
assert np.allclose(x.grad, np.ones_like(x.data))

x = Value(np.random.randn(4, 5))
x.grad[:] = 0 # reset gradient
y = x.sum(axis=1).mean()
y.backward()
assert np.allclose(x.grad, np.full_like(x.data, 1/4))