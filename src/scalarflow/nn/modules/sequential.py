from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module


class Sequential(Module):
    """A sequential container for neural network modules.

    Modules will be added to the sequential container in the order they are
    passed. The forward pass applies each module in sequence, passing the output
    of one module as input to the next.
    """

    def __init__(self, modules: list[Module]) -> None:
        """Initialise the sequential container.

        Args:
            modules: List of modules to apply in sequence.

        Raises:
            ValueError: If modules list is empty.
        """
        if not modules:
            raise ValueError(f"{self.__class__.__name__} requires at least one module")

        self.modules: list[Module] = list(modules)

    @override
    def __call__(self, inputs: list[Scalar]) -> list[Scalar]:
        """Forward pass through all modules in sequence.

        Args:
            inputs: List of input scalars.

        Returns:
            List of output scalars after passing through all modules.
        """
        outputs = inputs
        for module in self.modules:
            outputs = module(outputs)
        return outputs

    @override
    def parameters(self) -> list[Scalar]:
        """List trainable parameters from all contained modules.

        Returns:
            List containing all parameters from all modules in order.
        """
        return [p for module in self.modules for p in module.parameters()]
