import pytest

from scalarflow import Scalar
from scalarflow.nn import HuberLoss, Linear


def test__huber_loss__init_default() -> None:
    huber = HuberLoss()
    assert huber.delta == 1.0


def test__huber_loss__init_custom_delta() -> None:
    huber = HuberLoss(delta=2.5)
    assert huber.delta == 2.5


@pytest.mark.parametrize("delta", [0.0, -1.0])
def test__huber_loss__init_invalid_delta(delta: float) -> None:
    with pytest.raises(ValueError, match="delta must be positive"):
        HuberLoss(delta=delta)


def test__huber_loss__quadratic_region() -> None:
    huber = HuberLoss(delta=1.0)

    # Small errors should use quadratic loss: 0.5 * error^2
    predictions = [Scalar(1.5), Scalar(2.2)]
    targets = [Scalar(1.0), Scalar(2.0)]
    outputs = huber(predictions, targets)

    # Errors: |1.5-1.0| = 0.5, |2.2-2.0| = 0.2 (both <= delta=1.0)
    # Loss: (0.5*0.5^2 + 0.5*0.2^2) / 2 = (0.125 + 0.02) / 2 = 0.0725
    expected = (0.5 * 0.5**2 + 0.5 * 0.2**2) / 2
    assert len(outputs) == 1
    assert outputs[0].data == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def test__huber_loss__linear_region() -> None:
    huber = HuberLoss(delta=1.0)

    # Large errors should use linear loss: delta * |error| - 0.5 * delta^2
    predictions = [Scalar(3.0), Scalar(1.0)]
    targets = [Scalar(1.0), Scalar(4.0)]
    outputs = huber(predictions, targets)

    # Errors: |3.0-1.0| = 2.0, |1.0-4.0| = 3.0 (both > delta=1.0)
    # Loss: (1.0*2.0 - 0.5*1.0^2 + 1.0*3.0 - 0.5*1.0^2) / 2 = (1.5 + 2.5) / 2 = 2.0
    expected = (1.0 * 2.0 - 0.5 * 1.0**2 + 1.0 * 3.0 - 0.5 * 1.0**2) / 2
    assert len(outputs) == 1
    assert outputs[0].data == expected


def test__huber_loss__boundary_case() -> None:
    huber = HuberLoss(delta=1.0)

    # Error exactly at delta should give same result from both formulations
    predictions = [Scalar(2.0)]
    targets = [Scalar(1.0)]
    outputs = huber(predictions, targets)

    # Error: |2.0-1.0| = 1.0 = delta
    # Quadratic loss: 0.5 * 1.0^2 = 0.5
    # Linear loss: 1.0 * 1.0 - 0.5 * 1.0^2 = 0.5
    assert len(outputs) == 1
    assert outputs[0].data == 0.5


def test__huber_loss__mixed_regions() -> None:
    huber = HuberLoss(delta=1.0)

    predictions = [Scalar(1.5), Scalar(3.0)]
    targets = [Scalar(1.0), Scalar(1.0)]
    outputs = huber(predictions, targets)

    # Error 1: |1.5-1.0| = 0.5 <= delta => quadratic: 0.5 * 0.5^2 = 0.125
    # Error 2: |3.0-1.0| = 2.0 > delta => linear: 1.0 * 2.0 - 0.5 * 1.0^2 = 1.5
    expected = (0.5 * 0.5**2 + 1.0 * 2.0 - 0.5 * 1.0**2) / 2
    assert len(outputs) == 1
    assert outputs[0].data == expected


def test__huber_loss__perfect_predictions() -> None:
    huber = HuberLoss(delta=1.0)

    predictions = [Scalar(1.0), Scalar(2.0), Scalar(3.0)]
    targets = [Scalar(1.0), Scalar(2.0), Scalar(3.0)]
    outputs = huber(predictions, targets)

    assert len(outputs) == 1
    assert outputs[0].data == 0.0


def test__huber_loss__empty_inputs() -> None:
    huber = HuberLoss()

    predictions: list[Scalar] = []
    targets: list[Scalar] = []
    outputs = huber(predictions, targets)

    assert len(outputs) == 1
    assert outputs[0].data == 0.0


def test__huber_loss__input_validation() -> None:
    huber = HuberLoss()

    predictions = [Scalar(1.0), Scalar(2.0)]
    targets = [Scalar(1.0)]

    with pytest.raises(
        ValueError, match="Predictions and targets must have same length"
    ):
        huber(predictions, targets)


def test__huber_loss__parameters() -> None:
    huber = HuberLoss(delta=2.0)
    params = huber.parameters()

    assert len(params) == 0
    assert params == []


def test__huber_loss__gradient_flow_quadratic() -> None:
    huber = HuberLoss(delta=1.0)

    predictions = [Scalar(1.5), Scalar(2.2)]
    targets = [Scalar(1.0), Scalar(2.0)]

    loss = huber(predictions, targets)[0]
    loss.backward()

    # Both errors are in quadratic region
    # Error 1: 0.5, gradient = error/n = 0.5/2 = 0.25
    # Error 2: 0.2, gradient = error/n = 0.2/2 = 0.1
    assert predictions[0].grad == pytest.approx(0.25)  # pyright: ignore[reportUnknownMemberType]
    assert predictions[1].grad == pytest.approx(0.1)  # pyright: ignore[reportUnknownMemberType]


def test__huber_loss__gradient_flow_linear() -> None:
    huber = HuberLoss(delta=1.0)

    predictions = [Scalar(3.0), Scalar(-1.0)]
    targets = [Scalar(1.0), Scalar(1.0)]

    loss = huber(predictions, targets)[0]
    loss.backward()

    # Both errors are in linear region
    # Error 1: 2.0 > delta, gradient = sign(error) * delta / n = 1 * 1.0 / 2 = 0.5
    # Error 2: -2.0 < -delta, gradient = sign(error) * delta / n = -1 * 1.0 / 2 = -0.5
    assert predictions[0].grad == 0.5
    assert predictions[1].grad == -0.5


def test__huber_loss__gradient_flow_mixed() -> None:
    huber = HuberLoss(delta=1.0)

    predictions = [Scalar(1.5), Scalar(3.0)]
    targets = [Scalar(1.0), Scalar(1.0)]

    loss = huber(predictions, targets)[0]
    loss.backward()

    # Error 1: 0.5 <= delta, quadratic gradient = error/n = 0.5/2 = 0.25
    # Error 2: 2.0 > delta, linear gradient = sign(error) * delta / n = 0.5
    assert predictions[0].grad == 0.25
    assert predictions[1].grad == 0.5


def test__huber_loss__different_deltas() -> None:
    # Test that different delta values produce different results
    predictions = [Scalar(3.0)]
    targets = [Scalar(1.0)]

    huber_small = HuberLoss(delta=0.5)
    huber_large = HuberLoss(delta=2.0)

    loss_small = huber_small(predictions, targets)[0]
    loss_large = huber_large(predictions, targets)[0]

    # Small delta (0.5): linear loss = 0.5 * 2.0 - 0.5 * 0.5^2 = 0.875
    # Large delta (2.0): quadratic loss = 0.5 * 2.0^2 = 2.0
    expected_small = 0.5 * 2.0 - 0.5 * 0.5**2
    expected_large = 0.5 * 2.0**2

    assert loss_small.data == expected_small
    assert loss_large.data == expected_large
    assert loss_small.data != loss_large.data


def test__huber_loss__integration_with_linear() -> None:
    linear = Linear(1, 1, bias=False)
    huber = HuberLoss(delta=1.0)

    # Set deterministic weight
    linear.weights[0][0] = Scalar(2.0)

    # Forward pass
    inputs = [Scalar(3.0)]
    predictions = linear(inputs)  # 2 * 3 = 6
    targets = [Scalar(4.0)]
    loss = huber(predictions, targets)[0]

    # Error: |6-4| = 2 > delta=1.0
    # Linear loss: 1.0 * 2 - 0.5 * 1.0^2 = 1.5
    expected_loss = 1.0 * 2 - 0.5 * 1.0**2
    assert loss.data == expected_loss

    # Backward pass
    loss.backward()

    # Gradient in linear region: sign(error) * delta = 1 * 1.0 = 1.0
    # d(loss)/d(weight) = d(loss)/d(pred) * d(pred)/d(weight) = 1.0 * 3 = 3.0
    # d(loss)/d(input) = d(loss)/d(pred) * d(pred)/d(input) = 1.0 * 2 = 2.0
    assert linear.weights[0][0].grad == 3.0
    assert inputs[0].grad == 2.0
