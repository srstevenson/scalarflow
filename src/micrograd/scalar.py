from __future__ import annotations

import math
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class Scalar:
    def __init__(self, data: float, op: str = "", deps: Iterable[Scalar] = ()) -> None:
        self.data: float = data
        self.grad: float = 0.0
        self.op: str = op
        self.deps: frozenset[Scalar] = frozenset(deps)
        self._backward: Callable[[], None] = lambda: None

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.data:.4f}, grad={self.grad:.4f})"

    @classmethod
    def _coerce_other(cls, other: Scalar | float) -> Scalar:
        return other if isinstance(other, Scalar) else cls(other)

    def __pow__(self, other: Scalar | float) -> Scalar:
        other = self._coerce_other(other)
        result = Scalar(math.pow(self.data, other.data), "^", (self, other))

        def backward() -> None:
            self.grad += other.data * self.data ** (other.data - 1) * result.grad
            other.grad += result.data * math.log(self.data) * result.grad

        result._backward = backward
        return result

    def __add__(self, other: Scalar | float) -> Scalar:
        other = self._coerce_other(other)
        result = Scalar(self.data + other.data, "+", (self, other))

        def backward() -> None:
            self.grad += result.grad
            other.grad += result.grad

        result._backward = backward
        return result

    def __mul__(self, other: Scalar | float) -> Scalar:
        other = self._coerce_other(other)
        result = Scalar(self.data * other.data, "×", (self, other))

        def backward() -> None:
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = backward
        return result

    def __rpow__(self, other: Scalar | float) -> Scalar:
        other = self._coerce_other(other)
        return other**self

    def __radd__(self, other: float) -> Scalar:
        return self + other

    def __rmul__(self, other: float) -> Scalar:
        return self * other

    def __neg__(self) -> Scalar:
        return self * -1

    def __sub__(self, other: Scalar | float) -> Scalar:
        return self + -other

    def __rsub__(self, other: float) -> Scalar:
        return other + -self

    def __truediv__(self, other: Scalar | float) -> Scalar:
        return self * other**-1

    def __rtruediv__(self, other: float) -> Scalar:
        return other * self**-1

    def backward(self) -> None:
        self.grad = 1.0

        graph: dict[Scalar, frozenset[Scalar]] = {}
        stack: list[Scalar] = [self]
        while stack:
            node = stack.pop()
            graph[node] = node.deps
            stack.extend([dep for dep in node.deps if dep not in graph])

        ts: TopologicalSorter[Scalar] = TopologicalSorter(graph)
        for node in reversed(list(ts.static_order())):
            node._backward()  # noqa: SLF001

    def zero_grad(self) -> None:
        visited: set[Scalar] = set()
        stack: list[Scalar] = [self]
        while stack:
            if (node := stack.pop()) in visited:
                continue
            visited.add(node)
            node.grad = 0.0
            stack.extend(node.deps)

    def relu(self) -> Scalar:
        result = Scalar(max(0, self.data), "relu", (self,))

        def backward() -> None:
            self.grad += (self.data > 0) * result.grad

        result._backward = backward
        return result

    def tanh(self) -> Scalar:
        result = Scalar(math.tanh(self.data), "tanh", (self,))

        def backward() -> None:
            self.grad += (1 - result.data**2) * result.grad

        result._backward = backward
        return result

    def sigmoid(self) -> Scalar:
        result = Scalar(1 / (1 + math.exp(-self.data)), "sigmoid", (self,))

        def backward() -> None:
            self.grad += result.data * (1 - result.data) * result.grad

        result._backward = backward
        return result

    def exp(self) -> Scalar:
        result = Scalar(math.exp(self.data), "exp", (self,))

        def backward() -> None:
            self.grad += result.data * result.grad

        result._backward = backward
        return result

    def log(self) -> Scalar:
        result = Scalar(math.log(self.data), "log", (self,))

        def backward() -> None:
            self.grad += (1 / self.data) * result.grad

        result._backward = backward
        return result
