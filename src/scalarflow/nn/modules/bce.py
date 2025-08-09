from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module


class BCELoss(Module):
    """Binary cross entropy.

    Computes the mean binary cross entropy between predictions yp and targets
    y:

        -(1 / n) * sum(y * log(yp) + (1 - y) * log(1 - yp))

    Targets should be integers in {0, 1}. Predictions should be scalar
    probabilities in (0, 1). Inputs are clamped to a small epsilon range for
    numerical stability.
    """

    eps: float = 1e-7

    @override
    def __call__(self, preds: list[Scalar], targets: list[Scalar]) -> list[Scalar]:
        """Compute the binary cross entropy loss.

        Args:
            predictions: List of predicted probabilities (scalars in (0, 1)).
            targets: List of target labels (0 or 1) as scalars.

        Returns:
            A single-element list containing the mean BCE loss scalar.

        Raises:
            ValueError: If predictions and targets have different lengths.
        """
        if len(preds) != len(targets):
            raise ValueError(
                "Predictions and targets must have same length, "
                f"got {len(preds)} and {len(targets)}"
            )

        if len(preds) == 0:
            return [Scalar(0)]

        total = Scalar(0)
        for pred, target in zip(preds, targets, strict=True):
            p = pred.clamp(self.eps, 1 - self.eps)
            total += -(target * p.log() + (1 - target) * (1 - p).log())

        return [total / len(preds)]

    @override
    def parameters(self) -> list[Scalar]:
        """List trainable parameters in the module.

        Returns:
            list[Scalar]: Empty list as BCE loss has no trainable parameters.
        """
        return []
