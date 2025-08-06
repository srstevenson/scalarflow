import pytest

from scalarflow import Scalar
from scalarflow.nn import Linear, MSELoss


def test__mse_loss__forward_pass() -> None:
    mse = MSELoss()

    predictions = [Scalar(2.0), Scalar(3.0), Scalar(1.0)]
    targets = [Scalar(1.0), Scalar(3.0), Scalar(2.0)]
    outputs = mse(predictions, targets)

    # MSE = ((2-1)² + (3-3)² + (1-2)²) / 3 = (1 + 0 + 1) / 3 = 2/3
    assert len(outputs) == 1
    assert outputs[0].data == (2.0 / 3.0)


def test__mse_loss__perfect_predictions() -> None:
    mse = MSELoss()

    predictions = [Scalar(1.0), Scalar(2.0), Scalar(3.0)]
    targets = [Scalar(1.0), Scalar(2.0), Scalar(3.0)]
    outputs = mse(predictions, targets)

    assert len(outputs) == 1
    assert outputs[0].data == 0.0


def test__mse_loss__single_value() -> None:
    mse = MSELoss()

    predictions = [Scalar(5.0)]
    targets = [Scalar(2.0)]
    outputs = mse(predictions, targets)

    # MSE = (5-2)² / 1 = 9
    assert len(outputs) == 1
    assert outputs[0].data == 9.0


def test__mse_loss__empty_inputs() -> None:
    mse = MSELoss()

    predictions: list[Scalar] = []
    targets: list[Scalar] = []
    outputs = mse(predictions, targets)

    assert len(outputs) == 1
    assert outputs[0].data == 0.0


def test__mse_loss__input_validation() -> None:
    mse = MSELoss()

    predictions = [Scalar(1.0), Scalar(2.0)]
    targets = [Scalar(1.0)]

    with pytest.raises(
        ValueError, match="Predictions and targets must have same length"
    ):
        mse(predictions, targets)


def test__mse_loss__parameters() -> None:
    mse = MSELoss()
    params = mse.parameters()

    assert len(params) == 0
    assert params == []


def test__mse_loss__gradient_flow() -> None:
    mse = MSELoss()

    predictions = [Scalar(2.0), Scalar(4.0)]
    targets = [Scalar(1.0), Scalar(3.0)]

    loss = mse(predictions, targets)[0]
    loss.backward()

    # MSE = ((2-1)² + (4-3)²) / 2 = (1 + 1) / 2 = 1
    assert loss.data == 1.0

    # ∂MSE/∂pred₁ = 2(pred₁ - target₁) / n = 2(2-1) / 2 = 1
    # ∂MSE/∂pred₂ = 2(pred₂ - target₂) / n = 2(4-3) / 2 = 1
    assert predictions[0].grad == 1.0
    assert predictions[1].grad == 1.0


def test__mse_loss__integration_with_linear() -> None:
    linear = Linear(1, 1, bias=False)
    mse = MSELoss()

    # Set deterministic weight
    linear.weights[0][0] = Scalar(2.0)

    # Forward pass
    inputs = [Scalar(3.0)]
    predictions = linear(inputs)  # 2 * 3 = 6
    targets = [Scalar(4.0)]
    loss = mse(predictions, targets)[0]  # (6-4)² / 1 = 4

    assert loss.data == 4.0

    # Backward pass
    loss.backward()

    # ∂loss/∂weight = ∂loss/∂pred * ∂pred/∂weight = 4 * 3
    assert linear.weights[0][0].grad == 12.0
    # ∂loss/∂input = ∂loss/∂pred * ∂pred/∂input = 4 * 2
    assert inputs[0].grad == 8.0
