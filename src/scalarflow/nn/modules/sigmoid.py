from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module


class Sigmoid(Module):
    """Sigmoid activation function.

    Applies the sigmoid function 1 / (1 + exp(-x)) element-wise to input scalars
    x.
    """

    @override
    def __call__(self, inputs: list[Scalar]) -> list[Scalar]:
        """Apply sigmoid activation to input scalars.

        Args:
            inputs: List of input scalars.

        Returns:
            List of output scalars with sigmoid activation applied.
        """
        return [scalar.sigmoid() for scalar in inputs]

    @override
    def parameters(self) -> list[Scalar]:
        """List trainable parameters in the module.

        Returns:
            Empty list as sigmoid has no trainable parameters.
        """
        return []
