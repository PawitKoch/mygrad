import numpy as np
from .value import Value


def exp(x):
    x = x if isinstance(x, Value) else Value(x)
    out = Value(np.exp(x.data), (x,), 'exp')

    def _backward():
        x.grad += out.data * out.grad
    
    out._backward = _backward
    return out


def relu(x):
    x = x if isinstance(x, Value) else Value(x)
    out = Value(np.maximum(0, x.data), (x,), 'relu')

    def _backward():
        x.grad += (out.data > 0).astype(x.data.dtype) * out.grad

    out._backward = _backward
    return out


def sigmoid(x):
    x = x if isinstance(x, Value) else Value(x)
    sig = 1 / (1 + np.exp(-x.data))
    out = Value(sig, (x,), 'sigmoid')

    def _backward():
        x.grad += sig * (1 - sig) * out.grad
    
    out._backward = _backward
    return out


def bce_loss(pred, target, eps=1e-8):
    t = np.array(target.data if isinstance(target, Value) else target)
    p = pred.data

    loss = - (t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps))
    out = Value(loss.mean(), (pred,), 'bce_loss')
    
    def _backward():
        t = np.array(target.data if isinstance(target, Value) else target)
        grad = (-(t / (p + eps)) + (1 - t) / (1 - p + eps)) / t.size
        pred.grad += grad * out.grad
    
    out._backward = _backward
    return out


def softmax(x: Value):
    exps = np.exp(x.data - np.max(x.data, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    out = Value(probs, (x,), 'softmax')

    def _backward():
        pass # softmax handled directly in cross-entropy loss

    out._backward = _backward
    return out


def cross_entropy_loss(logits: Value, target: Value):
    '''
    logits: Value with shape (batch_size, num_classes)
    target: int array with shape (batch_size,) — ground-truth class indices
    '''
    probs = softmax(logits).data
    batch_size = probs.shape[0]

    log_likelihoods = -np.log(probs[np.arange(batch_size), target])
    loss_data = -np.mean(log_likelihoods)
    out = Value(loss_data, (logits,), 'xent')

    def _backward():
        grad = probs.copy()
        grad[np.arange(batch_size), target] -= 1
        grad /= batch_size
        logits.grad += grad * out.grad

    out._backward = _backward
    return out
