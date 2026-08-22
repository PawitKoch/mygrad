import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, model_dim, img_size, patch_size, n_channels):
        super().__init__()

        self.model_dim = model_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.conv = nn.Conv2d(self.n_channels, self.model_dim, kernel_size=self.patch_size, stride=self.patch_size)
    
    # P_col = Patch column, P_row = Patch row
    # P = P_col * P_row
    def forward(self, x):
        x = self.conv(x)  # (B, C, H, W) -> (B, model_dim, P_row, P_col)
        x = x.flatten(2)  # (B, model_dim, P_row, P_col) -> (B, model_dim, P)
        x = x.transpose(1, 2)  # (B, model_dim, P) -> (B, P, model_dim)
        return x

    
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))  # Classification token
        pos_encoding = torch.zeros(max_seq_len, model_dim)

        for pos in range(max_seq_len):
            for i in range(model_dim):
                if i % 2 == 0:
                    pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / model_dim)))
                else:
                    pos_encoding[pos, i] = np.cos(pos / (10000 ** ((i - 1) / model_dim)))
        
        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0))  # (1, max_seq_len, model_dim)
    
    def forward(self, x):
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)  # expand to have cls_token for each sample in the batch
        x = torch.cat((tokens_batch, x), dim=1)  # add cls_token to the beginning of each sequence
        x = x + self.pos_encoding  # (B, P+1, model_dim) + (1, max_seq_len, model_dim) -> (B, P+1, model_dim)
        return x


class AttentionHead(nn.Module):
    def __init__(self, model_dim, head_dim):
        super().__init__()

        self.head_dim = head_dim
        self.query = nn.Linear(model_dim, head_dim)
        self.key = nn.Linear(model_dim, head_dim)
        self.value = nn.Linear(model_dim, head_dim)

    def forward(self, x):
        Q = self.query(x)  # (B, P+1, model_dim) -> (B, P+1, head_dim)
        K = self.key(x)
        V = self.value(x)

        attention = Q @ K.transpose(-2, -1)
        attention = attention / (self.head_dim ** 0.5)
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


class ViT(nn.Module):
    def __init__(self, model_dim, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "Image dimensions must be divisible by the patch size."
        assert model_dim % n_heads == 0, "Model dimension must be divisible by the number of heads."

        self.model_dim = model_dim
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_heads = n_heads

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_len = self.n_patches + 1  # +1 for the cls_token

        self.patch_embedding = PatchEmbedding(self.model_dim, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEncoding(self.model_dim, self.max_seq_len)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(self.model_dim, self.n_heads) for _ in range(n_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim, self.n_classes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)  # (B, P, model_dim)
        x = self.positional_encoding(x)  # (B, P+1, model_dim)
        x = self.transformer_encoder(x)  # (B, P+1, model_dim)
        x = self.classifier(x[:, 0])  # Use the output corresponding to the cls_token for classification
        return x


def train_test(model_dim, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, batch_size, epochs, alpha):
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
    ])

    train_dataset = MNIST(root='data', train=True, transform=transform, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MNIST(root='data', train=False, transform=transform, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using device: ", device, f"({torch.backends.mps.get_name()})" if torch.backends.mps.is_available() else "")

    vit = ViT(model_dim, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
    optimizer = Adam(vit.parameters(), lr=alpha)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        training_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = vit(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(train_dataloader) :.3f}')
    

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = vit(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'\nModel Accuracy: {100 * correct // total} %')


if __name__ == "__main__":
    model_dim = 32
    n_classes = 10
    img_size = (64, 64)
    patch_size = (16, 16)
    n_channels = 1. # MNIST images are grayscale, so we have 1 channel
    n_heads = 8
    n_layers = 8
    batch_size = 128
    epochs = 15
    alpha = 0.001

    train_test(model_dim, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, batch_size, epochs, alpha)
