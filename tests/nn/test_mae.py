import pytest

from scalarflow import Scalar
from scalarflow.nn import Linear, MAELoss


def test__mae_loss__forward_pass() -> None:
    mae = MAELoss()

    predictions = [Scalar(2.0), Scalar(3.0), Scalar(1.0)]
    targets = [Scalar(1.0), Scalar(3.0), Scalar(2.0)]
    outputs = mae(predictions, targets)

    # MAE = (|2-1| + |3-3| + |1-2|) / 3 = (1 + 0 + 1) / 3 = 2/3
    assert len(outputs) == 1
    assert outputs[0].data == (2.0 / 3.0)


def test__mae_loss__perfect_predictions() -> None:
    mae = MAELoss()

    predictions = [Scalar(1.0), Scalar(2.0), Scalar(3.0)]
    targets = [Scalar(1.0), Scalar(2.0), Scalar(3.0)]
    outputs = mae(predictions, targets)

    assert len(outputs) == 1
    assert outputs[0].data == 0.0


def test__mae_loss__single_value() -> None:
    mae = MAELoss()

    predictions = [Scalar(5.0)]
    targets = [Scalar(2.0)]
    outputs = mae(predictions, targets)

    # MAE = |5-2| / 1 = 3
    assert len(outputs) == 1
    assert outputs[0].data == 3.0


def test__mae_loss__empty_inputs() -> None:
    mae = MAELoss()

    predictions: list[Scalar] = []
    targets: list[Scalar] = []
    outputs = mae(predictions, targets)

    assert len(outputs) == 1
    assert outputs[0].data == 0.0


def test__mae_loss__input_validation() -> None:
    mae = MAELoss()

    predictions = [Scalar(1.0), Scalar(2.0)]
    targets = [Scalar(1.0)]

    with pytest.raises(
        ValueError, match="Predictions and targets must have same length"
    ):
        mae(predictions, targets)


def test__mae_loss__parameters() -> None:
    mae = MAELoss()
    params = mae.parameters()

    assert len(params) == 0
    assert params == []


def test__mae_loss__gradient_flow() -> None:
    mae = MAELoss()

    predictions = [Scalar(2.0), Scalar(4.0)]
    targets = [Scalar(1.0), Scalar(3.0)]

    loss = mae(predictions, targets)[0]
    loss.backward()

    # MAE = (|2-1| + |4-3|) / 2 = (1 + 1) / 2 = 1
    assert loss.data == 1.0

    # ∂MAE/∂pred₁ = sign(pred₁ - target₁) / n = sign(2-1) / 2 = 1/2 = 0.5
    # ∂MAE/∂pred₂ = sign(pred₂ - target₂) / n = sign(4-3) / 2 = 1/2 = 0.5
    assert predictions[0].grad == 0.5
    assert predictions[1].grad == 0.5


def test__mae_loss__gradient_flow_negative_errors() -> None:
    mae = MAELoss()

    predictions = [Scalar(1.0), Scalar(3.0)]
    targets = [Scalar(2.0), Scalar(4.0)]

    loss = mae(predictions, targets)[0]
    loss.backward()

    # MAE = (|1-2| + |3-4|) / 2 = (1 + 1) / 2 = 1
    assert loss.data == 1.0

    # ∂MAE/∂pred₁ = sign(pred₁ - target₁) / n = sign(1-2) / 2 = -1/2 = -0.5
    # ∂MAE/∂pred₂ = sign(pred₂ - target₂) / n = sign(3-4) / 2 = -1/2 = -0.5
    assert predictions[0].grad == -0.5
    assert predictions[1].grad == -0.5


def test__mae_loss__gradient_flow_mixed_errors() -> None:
    mae = MAELoss()

    predictions = [Scalar(3.0), Scalar(1.0)]
    targets = [Scalar(1.0), Scalar(3.0)]

    loss = mae(predictions, targets)[0]
    loss.backward()

    # MAE = (|3-1| + |1-3|) / 2 = (2 + 2) / 2 = 2
    assert loss.data == 2.0

    # ∂MAE/∂pred₁ = sign(pred₁ - target₁) / n = sign(3-1) / 2 = 1/2 = 0.5
    # ∂MAE/∂pred₂ = sign(pred₂ - target₂) / n = sign(1-3) / 2 = -1/2 = -0.5
    assert predictions[0].grad == 0.5
    assert predictions[1].grad == -0.5


def test__mae_loss__integration_with_linear() -> None:
    linear = Linear(1, 1, bias=False)
    mae = MAELoss()

    # Set deterministic weight
    linear.weights[0][0] = Scalar(2.0)

    # Forward pass
    inputs = [Scalar(3.0)]
    predictions = linear(inputs)  # 2 * 3 = 6
    targets = [Scalar(4.0)]
    loss = mae(predictions, targets)[0]  # |6-4| / 1 = 2

    assert loss.data == 2.0

    # Backward pass
    loss.backward()

    # ∂loss/∂weight = ∂loss/∂pred * ∂pred/∂weight = 1 * 3 = 3
    assert linear.weights[0][0].grad == 3.0
    # ∂loss/∂input = ∂loss/∂pred * ∂pred/∂input = 1 * 2 = 2
    assert inputs[0].grad == 2.0
