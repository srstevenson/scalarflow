from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module


class MSELoss(Module):
    """Mean Squared Error (MSE) loss function module.

    Computes the mean squared error between predictions and targets:
    MSE = (1/n) * Σ(predictions[i] - targets[i])²
    """

    @override
    def __call__(
        self, predictions: list[Scalar], targets: list[Scalar]
    ) -> list[Scalar]:
        """Compute the mean squared error loss.

        Args:
            predictions: List of predicted scalars.
            targets: List of target scalars.

        Returns:
            Single-element list containing the MSE loss scalar.

        Raises:
            ValueError: If predictions and targets have different lengths.
        """
        if len(predictions) != len(targets):
            raise ValueError(
                "Predictions and targets must have same length, "
                f"got {len(predictions)} and {len(targets)}"
            )

        if len(predictions) == 0:
            return [Scalar(0.0)]

        sum_squared_error = sum(
            (
                (pred - target) ** 2
                for pred, target in zip(predictions, targets, strict=True)
            ),
            Scalar(0.0),
        )
        mse = sum_squared_error / len(predictions)

        return [mse]

    @override
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module.

        Returns:
            Empty list as MSE loss has no trainable parameters.
        """
        return []
