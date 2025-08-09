from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module


class Tanh(Module):
    """Tanh activation function.

    Applies the hyperbolic tangent (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    element-wise to input scalars x.
    """

    @override
    def __call__(self, inputs: list[Scalar]) -> list[Scalar]:
        """Apply tanh activation to input scalars.

        Args:
            inputs: List of input scalars.

        Returns:
            List of output scalars with tanh activation applied.
        """
        return [scalar.tanh() for scalar in inputs]

    @override
    def parameters(self) -> list[Scalar]:
        """List trainable parameters in the module.

        Returns:
            Empty list as tanh has no trainable parameters.
        """
        return []
