import numpy as np
from mygrad.core.value import Value
from mygrad.core.ops import cross_entropy_loss
from mygrad.nn.mlp import MLP
from mygrad.nn.optim import SGD
from mygrad.utils.metrics import classification_metrics


# X: 2D inputs
X = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.2],
        [0.2, 0.8],
    ]
)

# Y: class labels (0, 1, or 2)
Y = np.array([0, 1, 2, 1, 0, 2])

mlp = MLP(in_dim=2, hidden_dim=8, out_dim=3)
lr = 0.5
optimizer = SGD(mlp.parameters(), lr=lr)
epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()
    x = Value(X)
    y_pred = mlp(x)
    loss = cross_entropy_loss(y_pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

print("\nFinal predictions:")
y_preds = []
for x, y in zip(X, Y):
    x = Value(x.reshape(1, -1))
    y_pred = mlp(x)
    pred_class = y_pred.data.argmax(axis=1)[0]
    y_preds.append(pred_class)
    print(f"Input: {x.data}, Predicted: {pred_class}, Actual: {y}")

metrics = classification_metrics(Y, y_preds)
for c, stats in metrics.items():
    print(
        f"Class {c}: Precision: {stats['precision']:.3f}, Recall: {stats['recall']:.3f}, F1: {stats['f1']:.3f}"
    )
