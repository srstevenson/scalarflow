from abc import ABC, abstractmethod

from scalarflow import Scalar


class Optimiser(ABC):
    """Abstract base class for optimisers.

    This class defines the interface for all optimisation algorithms in
    ScalarFlow. Concrete optimiser implementations should inherit from this
    class and implement the required abstract methods.
    """

    def __init__(self, params: list[Scalar]) -> None:
        """Initialise the optimiser.

        Args:
            params: List of parameters to optimise.
        """
        self.params: list[Scalar] = params

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
        optimiser to zero, preparing for the next backward pass. It uses
        each parameter's zero_grad method to clear gradients throughout the
        computation graph.
        """
        for param in self.params:
            # It's important to call Scalar.zero_grad instead of setting
            # Scalar.grad to zero, so that we clear gradients for the entire
            # computation graph and not just the trainable parameters, which
            # would omit intermediate computations.
            param.zero_grad()
