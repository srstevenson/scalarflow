# ScalarFlow

A pedagogical automatic differentiation library with scalar primitives, inspired
by [micrograd](https://github.com/karpathy/micrograd).

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

Run `make` to see a list of all available commands.

[basedpyright]: https://docs.basedpyright.com/
[Graphviz]: https://graphviz.org/
[pytest]: https://docs.pytest.org/
[Ruff]: https://docs.astral.sh/ruff/
[uv]: https://docs.astral.sh/uv/
