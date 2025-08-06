from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module


class Sigmoid(Module):
    """Sigmoid activation function module.

    Applies the sigmoid function element-wise to input scalars.
    sigmoid(x) = 1 / (1 + e⁻ˣ).
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
        """Return all trainable parameters in the module.

        Returns:
            Empty list as sigmoid has no trainable parameters.
        """
        return []
