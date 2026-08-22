import numpy as np
from mygrad.nn.gpt import GPT
from mygrad.core.ops import cross_entropy_loss
from mygrad.nn.optim import SGD


with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()

# Tokenization
chars = sorted(set(text))
vocab_size = len(chars)

char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return np.array([char_to_id[c] for c in text])


def decode(ids):
    return "".join(id_to_char[i] for i in ids)


def get_batch(data, batch_size, seq_len):
    inputs = np.zeros((batch_size, seq_len), dtype=int)
    targets = np.zeros((batch_size, seq_len), dtype=int)
    for i in range(batch_size):
        start_idx = np.random.randint(0, len(data) - seq_len - 1)
        inputs[i] = data[start_idx : start_idx + seq_len]
        targets[i] = data[
            start_idx + 1 : start_idx + seq_len + 1
        ]  # shifted inputs by 1

    return inputs, targets


dataset = encode(text)
print(f"Vocab size: {vocab_size}, Dataset length: {len(dataset)}")

batch_size = 32
seq_len = 64
model = GPT(
    vocab_size=vocab_size, embed_dim=192, num_layers=4, num_heads=6, max_seq_len=64
)
base_lr = 0.01
optimizer = SGD(model.parameters(), lr=base_lr)
num_epochs = 20000
accum_steps = 4  # gradient accumulation steps

for epoch in range(num_epochs):
    inputs, targets = get_batch(dataset, batch_size, seq_len)
    optimizer.lr = (
        base_lr * 0.5 * (1 + np.cos(np.pi * epoch / num_epochs))
    )  # cosine decay

    logits = model(inputs)  # (batch_size, seq_len, vocab_size)
    logits_2d = logits.reshape((batch_size * seq_len, vocab_size))
    targets_1d = targets.flatten()  # (batch_size * seq_len,)

    loss = cross_entropy_loss(logits_2d, targets_1d)
    loss.backward()

    if (epoch + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}, LR: {optimizer.lr:.3f}")

# Final step for gradient accumulation
if (num_epochs) % accum_steps != 0:
    optimizer.step()
    optimizer.zero_grad()

print(f"Final Loss: {loss.data:.4f}, LR: {optimizer.lr:.3f}")
print("Saving model parameters to .npz")
params = [p.data for p in model.parameters()]
np.savez("gpt_shakespeare.npz", *params)
