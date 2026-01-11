import numpy as np
from mygrad.core.value import Value


def exp(x):
    x = x if isinstance(x, Value) else Value(x)
    out = Value(np.exp(x.data), (x,), "exp")

    def _backward():
        # Chain rule: if out = e^x, then:
        # ∂out/∂x = e^x (exponential derivative is itself!)
        # ∂L/∂x = ∂L/∂out * e^x = ∂L/∂out * out.data
        x.grad += out.data * out.grad

    out._backward = _backward
    return out


def relu(x):
    x = x if isinstance(x, Value) else Value(x)
    out = Value(np.maximum(0, x.data), (x,), "relu")

    def _backward():
        # Chain rule: if out = max(0, x) = ReLU(x), then:
        # ∂out/∂x = 1 if x > 0, else 0 (gradient is 0 for negative inputs)
        # ∂L/∂x = ∂L/∂out * ᵽ9(x > 0) where ᵽ9 is indicator function
        x.grad += (out.data > 0).astype(x.data.dtype) * out.grad

    out._backward = _backward
    return out


def sigmoid(x):
    x = x if isinstance(x, Value) else Value(x)
    sig = 1 / (1 + np.exp(-x.data))
    out = Value(sig, (x,), "sigmoid")

    def _backward():
        # Chain rule: if out = σ(x) = 1/(1 + e^(-x)), then:
        # ∂σ/∂x = σ(x) * (1 - σ(x)) (famous sigmoid derivative)
        # Proof: σ' = (1 + e^(-x))^(-2) * e^(-x) = σ * (1 - σ)
        # ∂L/∂x = ∂L/∂out * σ(x) * (1 - σ(x))
        x.grad += sig * (1 - sig) * out.grad

    out._backward = _backward
    return out


def bce_loss(pred, target, eps=1e-8):
    t = np.array(target.data if isinstance(target, Value) else target)
    p = pred.data

    loss = -(t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps))
    out = Value(loss.mean(), (pred,), "bce_loss")

    def _backward():
        # Chain rule: if L = -[t*log(p) + (1-t)*log(1-p)], then:
        # ∂L/∂p = -[t/p - (1-t)/(1-p)]
        #         = -(t/p) + (1-t)/(1-p)
        # This derivative tells us how to adjust predictions to minimize loss
        # eps added for numerical stability to avoid division by zero
        t = np.array(target.data if isinstance(target, Value) else target)
        grad = (-(t / (p + eps)) + (1 - t) / (1 - p + eps)) / t.size
        pred.grad += grad * out.grad

    out._backward = _backward
    return out


def softmax(x: Value, axis=-1):
    exps = np.exp(x.data - np.max(x.data, axis=axis, keepdims=True))
    probs = exps / np.sum(exps, axis=axis, keepdims=True)
    out = Value(probs, (x,), "softmax")

    def _backward():
        # Chain rule: if out_i = softmax(x)_i = e^x_i / Σ_j(e^x_j), then:
        # ∂out_i/∂x_j = out_i * (δ_ij - out_j) where δ_ij is Kronecker delta
        # In matrix form: ∂L/∂x = out ⊙ (∂L/∂out - (out · ∂L/∂out))
        # where ⊙ is element-wise product and · is dot product
        # Simplified: out * (dL/dout - sum(out * dL/dout))
        s_grad = out.grad * out.data
        sum_s_grad = np.sum(s_grad, axis=axis, keepdims=True)
        x.grad += s_grad - out.data * sum_s_grad

    out._backward = _backward
    return out


def cross_entropy_loss(logits: Value, target: Value):
    """
    logits: Value with shape (batch_size, num_classes)
    target: int array with shape (batch_size,) — ground-truth class indices
    """
    probs = softmax(logits).data
    batch_size = probs.shape[0]

    neg_log_likelihoods = -np.log(probs[np.arange(batch_size), target])
    loss_data = neg_log_likelihoods.mean()
    out = Value(loss_data, (logits,), "xent")

    def _backward():
        # Chain rule: Cross-entropy with softmax has elegant gradient!
        # If L = -log(softmax(x)_target), then:
        # ∂L/∂x_i = softmax(x)_i - δ_i,target
        # where δ_i,target = 1 if i == target else 0
        # In code: gradient is just (probabilities - one_hot(target)) / batch_size
        # This is why softmax + cross-entropy is popular: simple, stable gradient!
        grad = probs.copy()
        grad[np.arange(batch_size), target] -= 1  # Subtract 1 from true class
        grad /= batch_size
        logits.grad += grad * out.grad

    out._backward = _backward
    return out
