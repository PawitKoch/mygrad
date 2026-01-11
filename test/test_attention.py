import numpy as np
from mygrad.core.value import Value
from mygrad.nn.layers import MultiHeadAttention
from mygrad.core.utils import create_causal_mask


np.random.seed(42)
batch_size, seq_len, embed_dim = 2, 4, 8
num_heads = 4
x = Value(np.random.randn(batch_size, seq_len, embed_dim))

mha = MultiHeadAttention(embed_dim, num_heads)
out = mha(x)
assert out.data.shape == (batch_size, seq_len, embed_dim)
print(f"Output shape: {out.data.shape} ✓")

loss = out.sum()
loss.backward()

grads = [p.grad for p in mha.parameters()]
for i, g in enumerate(grads):
    print(f"Param {i} grad norm:", np.linalg.norm(g))

assert all(np.linalg.norm(g) > 0 for g in grads), (
    "All MHA parameters should have non-zero gradients."
)
print("✅ MultiHeadAttention test passed!\n")

np.random.seed(42)
batch_size, seq_len, embed_dim = 2, 4, 8
num_heads = 4
x = Value(np.random.randn(batch_size, seq_len, embed_dim))

# Create causal mask
causal_mask = create_causal_mask(seq_len)
print(f"Causal mask shape: {causal_mask.shape}")
print(f"Causal mask:\n{causal_mask}")

mha_causal = MultiHeadAttention(embed_dim, num_heads)
out_masked = mha_causal(x, mask=causal_mask)
assert out_masked.data.shape == (batch_size, seq_len, embed_dim)
print(f"\nOutput shape with mask: {out_masked.data.shape} ✓")

# Test backward pass with mask
loss_masked = out_masked.sum()
loss_masked.backward()

grads_masked = [p.grad for p in mha_causal.parameters()]
for i, g in enumerate(grads_masked):
    print(f"Param {i} grad norm: {np.linalg.norm(g):.6f}")

assert all(np.linalg.norm(g) > 0 for g in grads_masked), (
    "All MHA parameters should have non-zero gradients with mask."
)
print("✅ Causal MultiHeadAttention test passed!")
