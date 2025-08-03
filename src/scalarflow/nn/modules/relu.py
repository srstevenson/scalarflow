from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module


class ReLU(Module):
    """ReLU activation function module.

    Applies the Rectified Linear Unit function element-wise to input scalars.
    ReLU(x) = max(0, x).
    """

    @override
    def __call__(self, inputs: list[Scalar]) -> list[Scalar]:
        """Apply ReLU activation to input scalars.

        Args:
            inputs: List of input scalars.

        Returns:
            List of output scalars with ReLU activation applied.
        """
        return [scalar.relu() for scalar in inputs]

    @override
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module.

        Returns:
            Empty list as ReLU has no trainable parameters.
        """
        return []
