from abc import ABC, abstractmethod
from collections.abc import Sequence

from scalarflow import Scalar


class Optimiser(ABC):
    """Base class for optimisers.

    This class defines the interface for optimisation algorithms. Concrete
    implementations should inherit from this class and implement the abstract
    methods.
    """

    def __init__(self, params: Sequence[Scalar]) -> None:
        """Initialise the optimiser.

        Args:
            params: Sequence of parameters to optimise.
        """
        self.params: list[Scalar] = list(params)

    @abstractmethod
    def step(self) -> None:
        """Perform a single optimisation step to update parameters.

        This method should implement the core optimisation algorithm, updating
        the parameter values based on their gradients.
        """
        ...

    def zero_grad(self) -> None:
        """Clear gradients of all parameters.

        This method resets the gradients of all parameters managed by this
        optimiser to zero, preparing for the next backward pass.
        """
        for param in self.params:
            # Call Scalar.zero_grad instead of setting Scalar.grad to zero to
            # clear gradients for the entire computation graph and not just the
            # trainable parameters, which would omit intermediate computations.
            param.zero_grad()
