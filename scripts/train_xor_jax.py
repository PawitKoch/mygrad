import jax 
import jax.numpy as jnp
from jax import jit, value_and_grad


# XOR dataset
X = jnp.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

Y = jnp.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

def init_params(key):
    keys = jax.random.split(key, 4)
    W1 = jax.random.normal(keys[0], (2, 4)) * 0.1
    b1 = jnp.zeros((1, 4))
    W2 = jax.random.normal(keys[1], (4, 1)) * 0.1
    b2 = jnp.zeros((1, 1))

    return (W1, b1, W2, b2)

def forward(params, x):
    W1, b1, W2, b2 = params
    h = jnp.maximum(0, x @ W1 + b1) # ReLU activation
    out = jax.nn.sigmoid(h @ W2 + b2) # Sigmoid activation
    return out

def bce_loss(params, x, y, eps=1e-8):
    y_pred = forward(params, x)
    loss = - (y * jnp.log(y_pred + eps) + (1 - y) * jnp.log(1 - y_pred + eps))
    return jnp.mean(loss)

@jit
def update(params, x, y, lr):
    loss, grads = value_and_grad(bce_loss)(params, x, y)
    new_params = [param - lr * grad for param, grad in zip(params, grads)]
    return new_params, loss

def train(params, x, y, epochs=1000, lr=0.25):
    for epoch in range(epochs):
        params, loss = update(params, x, y, lr=lr)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")


# Train
key = jax.random.PRNGKey(42)
params = init_params(key)
lr = 0.25
epochs = 1000
train(params, X, Y, epochs, lr)

# Inference
print("\nFinal predictions:")
y_preds = forward(params, X)
for x, y, y_pred in zip(X, Y, y_preds):
    print(f"Input: {x}, Predicted: {y_pred.round(3).flatten()}, Actual: {y.flatten()}")
