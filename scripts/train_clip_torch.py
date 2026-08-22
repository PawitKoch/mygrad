import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len):
        super().__init__()

        pos_encoding = torch.zeros(max_seq_len, model_dim)

        for pos in range(max_seq_len):
            for i in range(model_dim):
                if i % 2 == 0:
                    pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / model_dim)))
                else:
                    pos_encoding[pos, i] = np.cos(pos / (10000 ** ((i - 1) / model_dim)))
        
        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0))  # (1, max_seq_len, model_dim)
    
    def forward(self, x):
        x += self.pos_encoding  # (B, P+1, model_dim) + (1, max_seq_len, model_dim) -> (B, P+1, model_dim)
        return x


class AttentionHead(nn.Module):
    def __init__(self, model_dim, head_dim):
        super().__init__()

        self.head_dim = head_dim
        self.query = nn.Linear(model_dim, head_dim)
        self.key = nn.Linear(model_dim, head_dim)
        self.value = nn.Linear(model_dim, head_dim)

    def forward(self, x, mask=None):
        Q = self.query(x)  # (B, P+1, model_dim) -> (B, P+1, head_dim)
        K = self.key(x)
        V = self.value(x)

        # Dot product attention
        attention = Q @ K.transpose(-2, -1)
        attention = attention / (self.head_dim ** 0.5)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(attention, dim=-1)
        attention = attention @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_heads):
        super().__init__()

        self.head_dim = model_dim // n_heads
        self.W_o = nn.Linear(model_dim, model_dim)
        self.heads = nn.ModuleList([AttentionHead(model_dim, self.head_dim) for _ in range(n_heads)])
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, P+1, model_dim)
        out = self.W_o(out)  # (B, P+1, model_dim) -> (B, P+1, model_dim)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, n_heads, r_mlp=4):
        super().__init__()
        
        self.model_dim = model_dim
        self.n_heads = n_heads

        # Sub-Layer 1: LayerNorm
        self.ln1 = nn.LayerNorm(model_dim)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(model_dim, n_heads)

        # Sub-Layer 2: LayerNorm
        self.ln2 = nn.LayerNorm(model_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, r_mlp * model_dim),
            nn.GELU(),
            nn.Linear(r_mlp * model_dim, model_dim)
        )

    def forward(self, x):
        out = x + self.mha(self.ln1(x))  # Residual connection around MHA
        out = out + self.mlp(self.ln2(out))  # Residual connection around MLP
        return out


def tokenizer(text, encode=True, mask=None, max_seq_len=32):
    if encode:
        out = chr(2) + text + chr(3) # Start and end tokens
        out = out + "".join([chr(0) for _ in range(max_seq_len - len(out))])  # Padding
        out = torch.IntTensor(list(out.encode("utf-8")))  # encoding text
        mask = torch.ones(len(out.nonzero()))
        mask = torch.cat((mask, torch.zeros(max_seq_len - len(mask)))).type(torch.IntTensor)
    else:
        out = [chr(x) for x in text[1:len(mask.nonzero()) - 1]]
        out = "".join(out)
        mask = None
    
    return out, mask