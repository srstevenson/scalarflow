# ScalarFlow

A pedagogical automatic differentiation library with scalar primitives, inspired
by [micrograd](https://github.com/karpathy/micrograd).

## Development

ScalarFlow uses [uv] for dependency management. Install the library and its
dependencies with:

```bash
uv sync
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

Format code using [Ruff]:

```bash
uv run ruff format src tests
```

Lint code using [Ruff]:

```bash
uv run ruff check --fix src tests
```

Type check using [basedpyright]:

```bash
uv run basedpyright src tests
```

Run the test suite using [pytest]:

```bash
uv run pytest tests
```

Measure test coverage with:

```bash
uv run coverage run -m pytest tests
uv run coverage report
```

[basedpyright]: https://docs.basedpyright.com/
[Graphviz]: https://graphviz.org/
[pytest]: https://docs.pytest.org/
[Ruff]: https://docs.astral.sh/ruff/
[uv]: https://docs.astral.sh/uv/
