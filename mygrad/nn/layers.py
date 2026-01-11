import numpy as np
import numpy.typing as npt
from mygrad.core.value import Value
from mygrad.core.ops import softmax, relu


class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = Value(np.random.randn(in_dim, out_dim) * 0.1)
        self.b = Value(np.zeros((1, out_dim)))

    def __call__(self, x: Value):
        out = x @ self.W
        return out + self.b

    def parameters(self):
        return [self.W, self.b]


class FeedForward:
    def __init__(self, embed_dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = embed_dim * 4
        self.l1 = Linear(embed_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, embed_dim)

    def __call__(self, x: Value):
        h = relu(self.l1(x))
        out = self.l2(h)
        return out

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()


class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = Value(np.random.randn(vocab_size, embed_dim) * 0.1)

    def __call__(self, indices: npt.NDArray):
        indices = np.array(indices, dtype=int)
        out = Value(self.W.data[indices], (self.W,), "embedding")

        def _backward():
            grad_W = np.zeros_like(self.W.data)
            np.add.at(grad_W, indices, out.grad)
            self.W.grad += grad_W

        out._backward = _backward
        return out

    def parameters(self):
        return [self.W]


class PositionalEncoding:
    def __init__(self, max_seq_len, embed_dim):
        # Learnable positional embeddings
        self.W = Value(np.random.randn(max_seq_len, embed_dim) * 0.02)

    def __call__(self, seq_len: int):
        # Return positional embeddings for sequence length
        return self.W[0:seq_len]

    def parameters(self):
        return [self.W]


class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = np.array(eps)  # convert to numpy scalar

        self.gamma = Value(np.ones(normalized_shape))  # gain/scale
        self.beta = Value(np.zeros(normalized_shape))  # bias/shift

    def __call__(self, x: Value):
        axis = -1  # always normalize over embed dim
        mean_val = x.data.mean(axis=axis, keepdims=True)
        variance_val = ((x.data - mean_val) ** 2).mean(axis=axis, keepdims=True)
        std_val = np.sqrt(variance_val + self.eps)
        x_norm_val = (x.data - mean_val) / std_val
        out_val = self.gamma.data * x_norm_val + self.beta.data
        out = Value(out_val, (x, self.gamma, self.beta), "layernorm")

        def _backward():
            # Gradient wrt beta: just sum incoming gradients over all but normalized dims
            # For shape (batch, seq, embed), we sum over (batch, seq) to get (embed,)
            sum_axes = tuple(range(len(x.data.shape) - len(self.normalized_shape)))
            self.beta.grad += out.grad.sum(axis=sum_axes)

            # Gradient wrt gamma: ∂L/∂γ = Σ(∂L/∂out * x_norm)
            self.gamma.grad += (out.grad * x_norm_val).sum(axis=sum_axes)

            # Gradient wrt x (the complex part!)
            # Formula: ∂L/∂x = (γ/σ) * [∂L/∂x_norm - mean(∂L/∂x_norm) - x_norm * mean(∂L/∂x_norm * x_norm)]

            # ∂L/∂x_norm = ∂L/∂out * γ
            grad_x_norm = out.grad * self.gamma.data

            # Compute means over normalized dimensions
            mean_grad = grad_x_norm.mean(axis=axis, keepdims=True)
            mean_grad_x_norm = (grad_x_norm * x_norm_val).mean(axis=axis, keepdims=True)

            # Final gradient
            grad_x = (1.0 / std_val) * (
                grad_x_norm - mean_grad - x_norm_val * mean_grad_x_norm
            )
            x.grad += grad_x

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]


class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.scale = np.sqrt(self.d_k)

        # Create linear layers for Q, K, V for all heads
        self.W_q = Linear(embed_dim, embed_dim)
        self.W_k = Linear(embed_dim, embed_dim)
        self.W_v = Linear(embed_dim, embed_dim)

        self.W_o = Linear(embed_dim, embed_dim)  # output projection

    def __call__(self, x: Value, mask=None):
        batch_size, seq_len, embed_dim = x.data.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.reshape((batch_size, seq_len, self.num_heads, self.d_k))
        Q = Q.transpose((0, 2, 1, 3))  # (batch, heads, seq_len, d_k)
        K = K.reshape((batch_size, seq_len, self.num_heads, self.d_k))
        K = K.transpose((0, 2, 1, 3))
        V = V.reshape((batch_size, seq_len, self.num_heads, self.d_k))
        V = V.transpose((0, 2, 1, 3))

        scores = (Q @ K.transpose((0, 1, 3, 2))) / self.scale
        if mask is not None:
            scores += mask

        weights = softmax(scores, axis=-1)
        out = weights @ V
        # Concat heads (batch, heads, seq_len, d_k) -> (batch, seq_len, heads * d_k)
        out = out.transpose((0, 2, 1, 3)).reshape((batch_size, seq_len, embed_dim))

        out = self.W_o(out)
        return out

    def parameters(self):
        return (
            self.W_q.parameters()
            + self.W_k.parameters()
            + self.W_v.parameters()
            + self.W_o.parameters()
        )


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, hidden_dim=None):
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)

        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.ln2 = LayerNorm(embed_dim)

    def __call__(self, x: Value, mask=None):
        # Residual connection + LayerNorm + MHA
        x = x + self.attn(self.ln1(x), mask=mask)

        # Residual connection + LayerNorm + FeedForward
        x = x + self.ffn(self.ln2(x))
        return x

    def parameters(self):
        return (
            self.attn.parameters()
            + self.ln1.parameters()
            + self.ffn.parameters()
            + self.ln2.parameters()
        )
