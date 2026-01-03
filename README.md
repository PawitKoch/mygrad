# `mygrad`

Welcome to **mygrad**! 🚀

This is a personal learning project where I implement my own automatic differentiation (autograd) engine from scratch using only **NumPy**. The goal is to better understand how frameworks like PyTorch and JAX work under the hood, and to benchmark my implementation against JAX for fun and learning! 🏋️‍♂️📈

## Features
- 🔢 Core autograd engine built with NumPy
- 🏗️ Simple neural network layers and optimizers
- 📊 Utilities for metrics and visualization
- 🤖 Example scripts for training on XOR and simple classification tasks
- ⚡ Benchmarking against JAX

## How to Use
1. Clone the repo
2. Install [uv](https://github.com/astral-sh/uv) if you haven't already
3. Sync dependencies: `uv sync`
4. Run the example scripts:
	- `uv run python scripts/train_xor_mygrad.py` (mygrad)
	- `uv run python scripts/train_xor_jax.py` (JAX)
   
   Or activate the virtual environment and run scripts directly:
	- `source .venv/bin/activate` (on Unix/macOS)
	- `python scripts/train_xor_mygrad.py`

## Folder Structure
- `mygrad/core/` – core autograd engine
- `mygrad/nn/` – neural network layers, MLP, optimizers
- `mygrad/utils/` – metrics and visualization
- Example scripts in the root directory

---

Made for fun and learning! 🧑‍💻✨
