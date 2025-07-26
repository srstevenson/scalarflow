from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Iterable


class Scalar:
    def __init__(self, data: float, children: Iterable[Scalar] = ()) -> None:
        self.data: float = data
        self.children: frozenset[Scalar] = frozenset(children)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.data})"

    def __neg__(self) -> Scalar:
        return -1 * self

    def __pow__(self, other: Scalar | float) -> Scalar:
        if isinstance(other, Scalar):
            return Scalar(math.pow(self.data, other.data), (self, other))
        return Scalar(math.pow(self.data, other), (self,))

    def __rpow__(self, other: float) -> Scalar:
        return Scalar(math.pow(other, self.data), (self,))

    def __add__(self, other: Scalar | float) -> Scalar:
        if isinstance(other, Scalar):
            return Scalar(self.data + other.data, (self, other))
        return Scalar(self.data + other, (self,))

    def __radd__(self, other: float) -> Scalar:
        return self + other

    def __mul__(self, other: Scalar | float) -> Scalar:
        if isinstance(other, Scalar):
            return Scalar(self.data * other.data, (self, other))
        return Scalar(self.data * other, (self,))

    def __rmul__(self, other: float) -> Scalar:
        return self * other

    def __sub__(self, other: Scalar | float) -> Scalar:
        if isinstance(other, Scalar):
            return Scalar(self.data - other.data, (self, other))
        return Scalar(self.data - other, (self,))

    def __rsub__(self, other: float) -> Scalar:
        return Scalar(other - self.data, (self,))

    def __truediv__(self, other: Scalar | float) -> Scalar:
        if isinstance(other, Scalar):
            return Scalar(self.data / other.data, (self, other))
        return Scalar(self.data / other, (self,))

    def __rtruediv__(self, other: float) -> Scalar:
        return Scalar(other / self.data, (self,))
