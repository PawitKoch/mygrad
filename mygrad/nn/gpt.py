from mygrad.nn.layers import (
    PositionalEncoding,
    TransformerBlock,
    LayerNorm,
    Linear,
    Embedding,
)
from mygrad.core.utils import create_causal_mask

import numpy.typing as npt


class GPT:
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len):
        self.token_embed = Embedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(max_seq_len, embed_dim)

        self.tf_blocks = [
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ]
        self.ln = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size)

    def __call__(self, token_ids: npt.NDArray):
        # token_ids: (batch_size, seq_len)
        seq_len = token_ids.shape[1]

        # Embed tokens + positional encodings
        x = self.token_embed(token_ids) + self.pos_embed(
            seq_len
        )  # (batch_size, seq_len, embed_dim)
        mask = create_causal_mask(seq_len)  # (seq_len, seq_len)

        for block in self.tf_blocks:
            x = block(x, mask=mask)

        x = self.ln(x)
        logits = self.head(x)  # (batch_size, seq_len, vocab_size)
        return logits

    def parameters(self):
        params = self.token_embed.parameters() + self.pos_embed.parameters()
        for block in self.tf_blocks:
            params += block.parameters()
        params += self.ln.parameters() + self.head.parameters()
        return params
