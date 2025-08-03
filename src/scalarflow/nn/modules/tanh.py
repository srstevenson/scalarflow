from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module


class Tanh(Module):
    """Tanh activation function module.

    Applies the hyperbolic tangent function element-wise to input scalars.
    Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)).
    """

    @override
    def __call__(self, inputs: list[Scalar]) -> list[Scalar]:
        """Apply Tanh activation to input scalars.

        Args:
            inputs: List of input scalars.

        Returns:
            List of output scalars with Tanh activation applied.
        """
        return [scalar.tanh() for scalar in inputs]

    @override
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module.

        Returns:
            Empty list as Tanh has no trainable parameters.
        """
        return []
