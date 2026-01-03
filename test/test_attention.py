import numpy as np
from mygrad.core.value import Value
from mygrad.nn.layers import SelfAttention


np.random.seed(42)
batch_size, seq_len, embed_dim = 2, 3, 4
x = Value(np.random.randn(batch_size, seq_len, embed_dim))

attn = SelfAttention(embed_dim)
out = attn(x)
assert out.data.shape == (batch_size, seq_len, embed_dim)

loss = out.sum()
loss.backward()

grads = [p.grad for p in attn.parameters()]
for i, g in enumerate(grads):
    print(f"Param {i} grad norm:", np.linalg.norm(g))

assert all(np.linalg.norm(g) > 0 for g in grads), "All parameters should have non-zero gradients."
