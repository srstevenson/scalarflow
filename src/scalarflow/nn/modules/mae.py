from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module


class MAELoss(Module):
    """Mean Absolute Error (MAE) loss function module.

    Computes the mean absolute error between predictions and targets:
    MAE = (1/n) × Σ|predictions[i] - targets[i]|
    """

    @override
    def __call__(
        self, predictions: list[Scalar], targets: list[Scalar]
    ) -> list[Scalar]:
        """Compute the mean absolute error loss.

        Args:
            predictions: List of predicted scalars.
            targets: List of target scalars.

        Returns:
            Single-element list containing the MAE loss scalar.

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

        sum_absolute_error = sum(
            (
                (pred - target).abs()
                for pred, target in zip(predictions, targets, strict=True)
            ),
            Scalar(0.0),
        )
        mae = sum_absolute_error / len(predictions)

        return [mae]

    @override
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module.

        Returns:
            Empty list as MAE loss has no trainable parameters.
        """
        return []
