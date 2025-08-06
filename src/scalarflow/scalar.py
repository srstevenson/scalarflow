"""Scalar-based automatic differentiation implementation.

This module provides the core Scalar class for automatic differentiation,
implementing a computational graph where each scalar tracks its value,
gradient, and dependencies. The implementation supports reverse-mode automatic
differentiation (backpropagation) with support for arithmetic operations,
mathematical functions, and commonly used activation functions.
"""

from __future__ import annotations

import math
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class Scalar:
    """A scalar value supporting automatic differentiation.

    This class implements a scalar-based automatic differentiation system, where
    each scalar tracks its value, gradient, and computational dependencies to
    enable reverse-mode automatic differentiation (backpropagation).

    Attributes:
        data: The scalar value stored in this node.
        grad: The gradient of some loss with respect to this scalar.
        op: A string describing the operation that created this scalar.
        deps: The set of scalars that this scalar depends on.
        backward_step: Function to compute gradients for this node's
            dependencies.
    """

    def __init__(self, data: float, op: str = "", deps: Iterable[Scalar] = ()) -> None:
        """Initialise a new scalar for automatic differentiation.

        Args:
            data: The scalar value to store.
            op: A string describing the operation that created this scalar.
            deps: An iterable of scalars that this scalar depends on.
        """
        self.data: float = data
        self.grad: float = 0.0
        self.op: str = op
        self.deps: frozenset[Scalar] = frozenset(deps)
        self.backward_step: Callable[[], None] = lambda: None

    @override
    def __repr__(self) -> str:
        """Return a string representation of the scalar.

        Returns:
            A string showing the scalar's data and gradient values.
        """
        return f"{self.__class__.__name__}(data={self.data:.4f}, grad={self.grad:.4f})"

    @classmethod
    def _coerce_other(cls, other: Scalar | float) -> Scalar:
        """Convert a float to a Scalar if needed.

        Args:
            other: A scalar or float value to potentially convert.

        Returns:
            A Scalar instance, either the original if already a Scalar,
            or a new Scalar created from the float value.
        """
        return other if isinstance(other, Scalar) else cls(other)

    def __pow__(self, other: Scalar | float) -> Scalar:
        """Raise this scalar to the power of another scalar or float.

        Args:
            other: The exponent value.

        Returns:
            A new Scalar representing self raised to the power of other.
        """
        other = self._coerce_other(other)
        result = Scalar(math.pow(self.data, other.data), "^", (self, other))

        def backward() -> None:
            self.grad += other.data * self.data ** (other.data - 1) * result.grad
            other.grad += result.data * math.log(self.data) * result.grad

        result.backward_step = backward
        return result

    def __add__(self, other: Scalar | float) -> Scalar:
        """Add this scalar to another scalar or float.

        Args:
            other: The value to add.

        Returns:
            A new Scalar representing the sum.
        """
        other = self._coerce_other(other)
        result = Scalar(self.data + other.data, "+", (self, other))

        def backward() -> None:
            self.grad += result.grad
            other.grad += result.grad

        result.backward_step = backward
        return result

    def __mul__(self, other: Scalar | float) -> Scalar:
        """Multiply this scalar by another scalar or float.

        Args:
            other: The value to multiply by.

        Returns:
            A new Scalar representing the product.
        """
        other = self._coerce_other(other)
        result = Scalar(self.data * other.data, "×", (self, other))

        def backward() -> None:
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result.backward_step = backward
        return result

    def __rpow__(self, other: Scalar | float) -> Scalar:
        """Compute other raised to the power of this scalar.

        Args:
            other: The base value.

        Returns:
            A new Scalar representing other raised to the power of self.
        """
        other = self._coerce_other(other)
        return other**self

    def __radd__(self, other: float) -> Scalar:
        """Add this scalar to a float (reverse addition).

        Args:
            other: The float value to add.

        Returns:
            A new Scalar representing the sum.
        """
        return self + other

    def __rmul__(self, other: float) -> Scalar:
        """Multiply this scalar by a float (reverse multiplication).

        Args:
            other: The float value to multiply by.

        Returns:
            A new Scalar representing the product.
        """
        return self * other

    def __neg__(self) -> Scalar:
        """Negate of this scalar.

        Returns:
            A new Scalar representing the negative of this scalar.
        """
        return self * -1

    def __sub__(self, other: Scalar | float) -> Scalar:
        """Subtract another scalar or float from this scalar.

        Args:
            other: The value to subtract.

        Returns:
            A new Scalar representing the difference.
        """
        return self + -other

    def __rsub__(self, other: float) -> Scalar:
        """Subtract this scalar from a float (reverse subtraction).

        Args:
            other: The float value to subtract from.

        Returns:
            A new Scalar representing the difference.
        """
        return other + -self

    def __truediv__(self, other: Scalar | float) -> Scalar:
        """Divide this scalar by another scalar or float.

        Args:
            other: The value to divide by.

        Returns:
            A new Scalar representing the quotient.
        """
        return self * other**-1

    def __rtruediv__(self, other: float) -> Scalar:
        """Divide a float by this scalar (reverse division).

        Args:
            other: The float value to divide.

        Returns:
            A new Scalar representing the quotient.
        """
        return other * self**-1

    def backward(self) -> None:
        """Compute gradients using reverse-mode automatic differentiation.

        This method initiates backpropagation from this scalar, computing
        gradients for all scalars in the computational graph that led to this
        scalar. The gradient computation uses topological sorting to ensure
        gradients are computed in the correct order.
        """
        self.grad = 1.0

        graph: dict[Scalar, frozenset[Scalar]] = {}
        stack: list[Scalar] = [self]
        while stack:
            node = stack.pop()
            graph[node] = node.deps
            stack.extend([dep for dep in node.deps if dep not in graph])

        ts: TopologicalSorter[Scalar] = TopologicalSorter(graph)
        for node in reversed(list(ts.static_order())):
            node.backward_step()

    def zero_grad(self) -> None:
        """Reset gradients to zero for this scalar and all its dependencies.

        This method traverses the computational graph starting from this scalar
        and sets the gradient of every scalar in the graph to zero. This is
        typically called before performing a new backward pass.
        """
        visited: set[Scalar] = set()
        stack: list[Scalar] = [self]
        while stack:
            if (node := stack.pop()) in visited:
                continue
            visited.add(node)
            node.grad = 0.0
            stack.extend(node.deps)

    def relu(self) -> Scalar:
        """Apply the ReLU function.

        Returns:
            A new Scalar with ReLU applied: max(0, self).
        """
        result = Scalar(max(0, self.data), "relu", (self,))

        def backward() -> None:
            self.grad += (self.data > 0) * result.grad

        result.backward_step = backward
        return result

    def tanh(self) -> Scalar:
        """Apply the hyperbolic tangent function.

        Returns:
            A new Scalar with tanh applied to this scalar's value.
        """
        result = Scalar(math.tanh(self.data), "tanh", (self,))

        def backward() -> None:
            self.grad += (1 - result.data**2) * result.grad

        result.backward_step = backward
        return result

    def sigmoid(self) -> Scalar:
        """Apply the sigmoid function.

        Returns:
            A new Scalar with sigmoid applied: 1 / (1 + e⁻ˣ).
        """
        result = Scalar(1 / (1 + math.exp(-self.data)), "sigmoid", (self,))

        def backward() -> None:
            self.grad += result.data * (1 - result.data) * result.grad

        result.backward_step = backward
        return result

    def exp(self) -> Scalar:
        """Apply the exponential function.

        Returns:
            A new Scalar representing e raised to the power of this scalar.
        """
        result = Scalar(math.exp(self.data), "exp", (self,))

        def backward() -> None:
            self.grad += result.data * result.grad

        result.backward_step = backward
        return result

    def log(self) -> Scalar:
        """Apply the natural logarithm function.

        Returns:
            A new Scalar representing the natural logarithm of this scalar.
        """
        result = Scalar(math.log(self.data), "log", (self,))

        def backward() -> None:
            self.grad += (1 / self.data) * result.grad

        result.backward_step = backward
        return result

    def sqrt(self) -> Scalar:
        """Apply the square root function.

        Returns:
            A new Scalar representing the square root of this scalar.
        """
        result = Scalar(math.sqrt(self.data), "sqrt", (self,))

        def backward() -> None:
            self.grad += (1 / (2 * result.data)) * result.grad

        result.backward_step = backward
        return result

    def abs(self) -> Scalar:
        """Apply the absolute value function.

        Returns:
            A new Scalar representing the absolute value of this scalar.
        """
        result = Scalar(abs(self.data), "abs", (self,))

        def backward() -> None:
            if self.data > 0:
                self.grad += 1 * result.grad
            elif self.data < 0:
                self.grad += -1 * result.grad
            else:
                # Any value in [-1, 1] is valid for the gradient at 0. We follow
                # PyTorch in using 0, whereas JAX uses 1.
                self.grad += 0 * result.grad

        result.backward_step = backward
        return result

    def min(self, other: Scalar | float) -> Scalar:
        """Compute the minimum of this scalar and another value.

        Args:
            other: The value to compare with.

        Returns:
            A new Scalar representing the minimum of self and other.
        """
        other = self._coerce_other(other)
        result = Scalar(min(self.data, other.data), "min", (self, other))

        def backward() -> None:
            if self.data < other.data:
                self.grad += result.grad
            elif self.data > other.data:
                other.grad += result.grad
            else:
                # Split gradient equally when values are tied.
                self.grad += 0.5 * result.grad
                other.grad += 0.5 * result.grad

        result.backward_step = backward
        return result

    def max(self, other: Scalar | float) -> Scalar:
        """Compute the maximum of this scalar and another value.

        Args:
            other: The value to compare with.

        Returns:
            A new Scalar representing the maximum of self and other.
        """
        other = self._coerce_other(other)
        result = Scalar(max(self.data, other.data), "max", (self, other))

        def backward() -> None:
            if self.data > other.data:
                self.grad += result.grad
            elif self.data < other.data:
                other.grad += result.grad
            else:
                # Split gradient equally when values are tied.
                self.grad += 0.5 * result.grad
                other.grad += 0.5 * result.grad

        result.backward_step = backward
        return result

    def sin(self) -> Scalar:
        """Apply the sine function.

        Returns:
            A new Scalar representing the sine of this scalar.
        """
        result = Scalar(math.sin(self.data), "sin", (self,))

        def backward() -> None:
            self.grad += math.cos(self.data) * result.grad

        result.backward_step = backward
        return result

    def cos(self) -> Scalar:
        """Apply the cosine function.

        Returns:
            A new Scalar representing the cosine of this scalar.
        """
        result = Scalar(math.cos(self.data), "cos", (self,))

        def backward() -> None:
            self.grad += -math.sin(self.data) * result.grad

        result.backward_step = backward
        return result

    def clamp(
        self, min_val: float | None = None, max_val: float | None = None
    ) -> Scalar:
        """Clamp this scalar's value between optional min and max bounds.

        Args:
            min_val: The minimum value to clamp to. If None, no minimum is
                applied.
            max_val: The maximum value to clamp to. If None, no maximum is
                applied.

        Returns:
            A new Scalar with the value clamped between the specified bounds.

        Raises:
            ValueError: If min_val is greater than max_val.
        """
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")

        result = self
        if min_val is not None:
            result = result.max(min_val)
        if max_val is not None:
            result = result.min(max_val)
        return result
