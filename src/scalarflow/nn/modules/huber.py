from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module


class HuberLoss(Module):
    """Huber loss function module.

    Combines quadratic loss for small errors and linear loss for large errors:
    L_δ(y, ŷ) = {
        0.5 * (y - ŷ)²           if |y - ŷ| ≤ δ
        δ * |y - ŷ| - 0.5 * δ²   if |y - ŷ| > δ
    }

    This provides smooth gradients near zero (like MSE) while being robust
    to outliers (like MAE).
    """

    def __init__(self, delta: float = 1.0) -> None:
        """Initialise the Huber loss with the given delta parameter.

        Args:
            delta: The threshold at which to switch from quadratic to linear
                loss. Must be positive.

        Raises:
            ValueError: If delta is not positive.
        """
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.delta: float = delta

    @override
    def __call__(
        self, predictions: list[Scalar], targets: list[Scalar]
    ) -> list[Scalar]:
        """Compute the Huber loss.

        Args:
            predictions: List of predicted scalars.
            targets: List of target scalars.

        Returns:
            Single-element list containing the Huber loss scalar.

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

        sum_huber_loss = sum(
            (
                self._huber_loss_single(pred, target)
                for pred, target in zip(predictions, targets, strict=True)
            ),
            Scalar(0.0),
        )
        huber = sum_huber_loss / len(predictions)

        return [huber]

    def _huber_loss_single(self, prediction: Scalar, target: Scalar) -> Scalar:
        """Compute Huber loss for a single prediction-target pair.

        Args:
            prediction: The predicted scalar.
            target: The target scalar.

        Returns:
            The Huber loss for this pair.
        """
        error = prediction - target
        abs_error = error.abs()

        # Use the quadratic loss when |error| <= δ, and add the linear excess
        # when |error| > δ i.e.
        # min(0.5*error², 0.5*δ²) + δ*max(0, |error| - δ)
        quadratic_part = 0.5 * error * error
        clamped_quadratic = quadratic_part.min(0.5 * self.delta**2)

        linear_part = self.delta * (abs_error - self.delta).max(0)

        return clamped_quadratic + linear_part

    @override
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module.

        Returns:
            Empty list as Huber loss has no trainable parameters.
        """
        return []
