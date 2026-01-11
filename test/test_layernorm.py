import numpy as np
from mygrad.core.value import Value
from mygrad.nn.layers import LayerNorm

np.random.seed(42)

# Test LayerNorm forward pass
batch_size, seq_len, embed_dim = 2, 4, 8
x = Value(np.random.randn(batch_size, seq_len, embed_dim))

ln = LayerNorm(embed_dim)
out = ln(x)

print("Input shape:", x.data.shape)
print("Output shape:", out.data.shape)

# Check normalization properties: mean ≈ 0, std ≈ 1 (before gamma/beta)
# Since gamma=1 and beta=0 initially, output should be normalized
mean = out.data.mean(axis=-1)
std = out.data.std(axis=-1)
print("\nOutput mean (should be close to 0):")
print(mean)
print("\nOutput std (should be close to 1):")
print(std)

# Test backward pass
loss = (out * Value(np.random.randn(*out.data.shape))).sum()  # Non-uniform gradients
loss.backward()

print("\n" + "=" * 50)
print("Gradient check:")
print("=" * 50)

grads = [p.grad for p in ln.parameters()]
print(f"Gamma grad norm: {np.linalg.norm(grads[0]):.6f}")
print(f"Beta grad norm: {np.linalg.norm(grads[1]):.6f}")
print(f"Input grad norm: {np.linalg.norm(x.grad):.15f}")  # More precision
print(f"Input grad min/max: {x.grad.min():.10f} / {x.grad.max():.10f}")
print(f"Input grad mean: {x.grad.mean():.15f}")
print(f"Input grad std: {x.grad.std():.15f}")

# Check if backward was called
print(f"\nChecking if x is in out._prev: {x in out._prev}")
print(f"out._prev: {[id(p) for p in out._prev]}")
print(f"x id: {id(x)}")

assert all(np.linalg.norm(g) > 0 for g in grads), (
    "All parameters should have non-zero gradients"
)
assert np.linalg.norm(x.grad) > 0, "Input should have non-zero gradient"

print("\n✅ All tests passed!")
