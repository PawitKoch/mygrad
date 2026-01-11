import numpy as np
from mygrad.nn.layers import Embedding

# Toy vocabulary and sequence
vocab_size = 5
embedding_dim = 3
seq_length = 4
batch_size = 2

# Input indices
X = np.array([[0, 1, 2, 3], [3, 2, 1, 4]])

print("Input shape:", X.shape)

embedding = Embedding(vocab_size, embedding_dim)

# Forward pass
embeds = embedding(X)  # shape: (batch_size, seq_length, embedding_dim)

print("Embedding output shape:", embeds.data.shape)
print("Embedding output:\n", embeds.data)

# Backward pass (dummy loss: sum of all embedding entries)
loss = embeds.sum()
loss.backward()

W = embedding.parameters()[0]
print("\nEmbedding gradients shape:", W.grad.shape)
print("Embedding gradients (nonzero for used indices):\n", W.grad)
