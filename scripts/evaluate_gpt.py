import numpy as np

from mygrad.nn.gpt import GPT


# Tokenization
with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return np.array([char_to_id[c] for c in text])


def decode(ids):
    return "".join(id_to_char[i] for i in ids)


def generate(model, seq_len, start_prompt, max_new_tokens=100):
    context = encode(start_prompt)

    for _ in range(max_new_tokens):
        # Get logits for the current context
        ctx = (
            context[-seq_len:] if len(context) > seq_len else context
        )  # ensure context length <= seq_len
        logits = model(ctx.reshape(1, -1))  # (1, seq_len, vocab_size)

        # Next token prediction using sampling
        next_token_logits = logits[:, -1, :].data  # (1, vocab_size)
        probs = np.exp(
            next_token_logits[0] - np.max(next_token_logits[0])
        )  # for numerical stability
        probs = probs / probs.sum()
        next_token = np.random.choice(len(probs), p=probs)

        # Append to context
        context = np.append(context, next_token)

    return decode(context)


# Init model and load trained parameters
model = GPT(
    vocab_size=vocab_size,
    embed_dim=192,
    num_layers=4,
    num_heads=6,
    max_seq_len=64,
)
params = np.load("gpt_shakespeare.npz")
for p, arr in zip(model.parameters(), params.values()):
    p.data = arr

# Generate text
print("\n--- Generated Text ---")
start_prompt = "ROMEO:\n"
print(generate(model, seq_len=64, start_prompt=start_prompt))
