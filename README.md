<p align="center"><img src="assets/scalarflow.png" width=300></p>

ScalarFlow is a pedagogical deep learning library with scalar primitives,
inspired by [micrograd] and [PyTorch]. It has no dependencies outside the Python
standard library, except for the optional [Graphviz] package used for
visualisation.

## Overview

ScalarFlow is built around a `Scalar` class that enables automatic
differentiation. The library also includes a neural network module
(`scalarflow.nn`) and an optimisation module (`scalarflow.optim`), both modelled
on PyTorch's API. The neural network module provides a `Module` base class and
common layers like `Linear`, `ReLU`, `Tanh`, `MSELoss`, and `Sequential`. The
optimisation module includes optimisers like `SGD` for training neural networks.

## Usage

The following snippet shows how to use ScalarFlow to create a model, set up an
optimiser, and perform a complete training step:

```python
from scalarflow import Scalar, nn, optim

model = nn.Sequential([nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1)])
loss_fn = nn.MSELoss()
optimiser = optim.SGD(model.parameters(), lr=0.01)

x = [Scalar(2.0)]
y = [Scalar(4.0)]

# Forward pass
y_hat = model(x)
loss = loss_fn(y_hat, y)[0]

# Backward pass
loss.backward()

# Update parameters
optimiser.step()
optimiser.zero_grad()

print(f"Loss: {loss.data:.4f}")
```

For a more complete example, see
[`examples/quadratic.py`](examples/quadratic.py), which trains a model on a
dataset of random data.

```
$ uv run examples/quadratic.py
Epoch | Train Loss | Train RMSE | Val Loss | Val RMSE
----- | ---------- | ---------- | -------- | --------
    0 |     0.3220 |     0.5674 |   0.1879 |   0.4335
  100 |     0.0768 |     0.2771 |   0.0429 |   0.2072
  200 |     0.0699 |     0.2643 |   0.0311 |   0.1764
  300 |     0.0663 |     0.2575 |   0.0285 |   0.1687
  400 |     0.0603 |     0.2456 |   0.0253 |   0.1590
  500 |     0.0527 |     0.2296 |   0.0227 |   0.1506
  600 |     0.0445 |     0.2109 |   0.0202 |   0.1423
  700 |     0.0359 |     0.1896 |   0.0140 |   0.1184
  800 |     0.0258 |     0.1607 |   0.0078 |   0.0881
  900 |     0.0190 |     0.1379 |   0.0048 |   0.0693
 1000 |     0.0144 |     0.1199 |   0.0028 |   0.0528
```

## Development

ScalarFlow uses [uv] for dependency management. Install the library and its
dependencies with:

```bash
make install
```

ScalarFlow requires [Graphviz] for visualisation functionality. Install Graphviz
separately using your system package manager:

```bash
# macOS
brew install graphviz

# Arch Linux
pacman -Syu graphviz

# Debian/Ubuntu
apt install graphviz
```

Format with [Ruff], lint with Ruff and [basedpyright], and test with [pytest]
using:

```bash
make all
```

Run `make help` to see a list of all available commands.

[basedpyright]: https://docs.basedpyright.com/
[Graphviz]: https://graphviz.org/
[micrograd]: https://github.com/karpathy/micrograd
[pytest]: https://docs.pytest.org/
[PyTorch]: https://pytorch.org
[Ruff]: https://docs.astral.sh/ruff/
[uv]: https://docs.astral.sh/uv/
