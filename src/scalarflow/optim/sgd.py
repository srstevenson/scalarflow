from typing import override

from scalarflow import Scalar
from scalarflow.optim.base import Optimiser


class SGD(Optimiser):
    """Stochastic Gradient Descent optimiser.

    This class implements the standard stochastic gradient descent algorithm
    with a fixed learning rate.
    """

    def __init__(self, params: list[Scalar], lr: float) -> None:
        """Initialise the SGD optimiser.

        Args:
            params: List of parameters to optimise.
            lr: Learning rate for gradient descent.

        Raises:
            ValueError: If learning rate is not positive.
        """
        super().__init__(params)

        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")

        self.lr: float = lr

    @override
    def step(self) -> None:
        """Perform a single SGD optimisation step."""
        for param in self.params:
            param.data -= self.lr * param.grad
