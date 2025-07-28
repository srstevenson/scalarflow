from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Iterable


class Scalar:
    def __init__(self, data: float, op: str = "", deps: Iterable[Scalar] = ()) -> None:
        self.data: float = data
        self.op: str = op
        self.deps: frozenset[Scalar] = frozenset(deps)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.data})"

    def __pow__(self, other: Scalar | float) -> Scalar:
        other = Scalar(other) if not isinstance(other, Scalar) else other
        return Scalar(math.pow(self.data, other.data), "^", (self, other))

    def __add__(self, other: Scalar | float) -> Scalar:
        other = Scalar(other) if not isinstance(other, Scalar) else other
        return Scalar(self.data + other.data, "+", (self, other))

    def __mul__(self, other: Scalar | float) -> Scalar:
        other = Scalar(other) if not isinstance(other, Scalar) else other
        return Scalar(self.data * other.data, "×", (self, other))

    def __rpow__(self, other: Scalar | float) -> Scalar:
        other = Scalar(other) if not isinstance(other, Scalar) else other
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
