.DEFAULT_GOAL := help

.PHONY: all
all: fmt lint test

.PHONY: install
install:
	uv sync

.PHONY: fmt
fmt:
	uv run ruff format src tests examples

.PHONY: lint
lint:
	uv run ruff check --fix src tests examples
	uv run basedpyright src tests examples

.PHONY: check
check:
	uv run ruff format --check src tests examples
	uv run ruff check src tests examples
	uv run basedpyright src tests examples

.PHONY: test
test:
	uv run coverage run -m pytest tests
	uv run coverage report

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make help     - Show this help message (default)"
	@echo "  make all      - Format (ruff), lint (ruff, basedpyright), and test (pytest)"
	@echo "  make install  - Install dependencies with uv"
	@echo "  make fmt      - Format code with ruff"
	@echo "  make lint     - Lint with ruff (--fix) and type-check with basedpyright"
	@echo "  make check    - Check formatting (ruff), lint (ruff), and types (basedpyright)"
	@echo "  make test     - Run tests with pytest and report coverage"
