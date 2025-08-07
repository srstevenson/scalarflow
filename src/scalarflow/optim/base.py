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

        This method should implement the core optimisation algorithm,
        updating the parameter values based on their gradients.
        """
        ...

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear gradients of all parameters.

        This method should reset the gradients of all parameters managed
        by this optimiser to zero, preparing for the next backward pass.
        """
        ...
