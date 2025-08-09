import math

import pytest

from scalarflow import Scalar
from scalarflow.nn import BCELoss


def _mean_bce(preds: list[float], targets: list[float]) -> float:
    """Mean BCE for floats for test assertions independent of Scalar version."""
    eps = 1e-7
    total = 0.0
    for prob, y in zip(preds, targets, strict=True):
        q = min(max(prob, eps), 1 - eps)
        total += -(y * math.log(q) + (1 - y) * math.log(1 - q))
    return total / len(preds) if preds else 0.0


def test__bce_loss__forward_pass() -> None:
    bce = BCELoss()

    predictions = [Scalar(0.9), Scalar(0.2), Scalar(0.7)]
    targets = [Scalar(1.0), Scalar(0.0), Scalar(1.0)]
    outputs = bce(predictions, targets)

    expected = _mean_bce([0.9, 0.2, 0.7], [1.0, 0.0, 1.0])
    assert len(outputs) == 1
    assert outputs[0].data == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def test__bce_loss__single_value() -> None:
    bce = BCELoss()

    predictions = [Scalar(0.8)]
    targets = [Scalar(1.0)]
    outputs = bce(predictions, targets)

    expected = _mean_bce([0.8], [1.0])
    assert len(outputs) == 1
    assert outputs[0].data == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def test__bce_loss__empty_inputs() -> None:
    bce = BCELoss()

    predictions: list[Scalar] = []
    targets: list[Scalar] = []
    outputs = bce(predictions, targets)

    assert len(outputs) == 1
    assert outputs[0].data == 0.0


def test__bce_loss__input_validation() -> None:
    bce = BCELoss()

    predictions = [Scalar(0.7), Scalar(0.3)]
    targets = [Scalar(1.0)]

    with pytest.raises(
        ValueError, match="Predictions and targets must have same length"
    ):
        bce(predictions, targets)


def test__bce_loss__parameters() -> None:
    bce = BCELoss()
    params = bce.parameters()

    assert params == []


def test__bce_loss__backward() -> None:
    bce = BCELoss()

    predictions = [Scalar(0.8), Scalar(0.3)]
    targets = [Scalar(1.0), Scalar(0.0)]

    loss = bce(predictions, targets)[0]
    loss.backward()

    assert loss.data == pytest.approx(_mean_bce([0.8, 0.3], [1.0, 0.0]))  # pyright: ignore[reportUnknownMemberType]

    # dL/dp = (-(y/p) + (1-y)/(1-p)) / n
    grad1 = (-(1.0 / 0.8) + 0.0) / 2.0
    grad2 = (-(0.0) + 1.0 / (1.0 - 0.3)) / 2.0
    assert predictions[0].grad == pytest.approx(grad1)  # pyright: ignore[reportUnknownMemberType]
    assert predictions[1].grad == pytest.approx(grad2)  # pyright: ignore[reportUnknownMemberType]
