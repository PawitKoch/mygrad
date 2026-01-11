import numpy as np
from mygrad.core.value import Value
from mygrad.core.ops import bce_loss
from mygrad.nn.mlp import MLP
from mygrad.nn.optim import SGD
from mygrad.utils.viz import plot_decision_boundary


# XOR dataset
X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)

Y = np.array(
    [
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ]
)

mlp = MLP(in_dim=2, hidden_dim=4, out_dim=1)
lr = 0.25
optimizer = SGD(mlp.parameters(), lr=lr)
epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()
    x = Value(X)
    y_pred = mlp(x)
    loss = bce_loss(y_pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")


print("\nFinal predictions:")
for x, y in zip(X, Y):
    x = Value(x.reshape(1, -1))
    y = Value(y.reshape(1, -1))
    y_pred = mlp(x)
    print(
        f"Input: {x.data}, Predicted: {y_pred.data.round(3).flatten()}, Actual: {y.data.flatten()}"
    )

plot_decision_boundary(mlp, X, Y)
