import math

from scalarflow import Scalar
from scalarflow.nn import Linear, Tanh


def test__tanh__forward_pass() -> None:
    tanh = Tanh()

    inputs = [Scalar(-2.0), Scalar(-1.0), Scalar(0.0), Scalar(1.0), Scalar(2.0)]
    outputs = tanh(inputs)

    assert len(outputs) == len(inputs)
    assert outputs[0].data == math.tanh(-2.0)
    assert outputs[1].data == math.tanh(-1.0)
    assert outputs[2].data == math.tanh(0.0)  # Should be 0.0
    assert outputs[3].data == math.tanh(1.0)
    assert outputs[4].data == math.tanh(2.0)


def test__tanh__empty_input() -> None:
    tanh = Tanh()

    inputs: list[Scalar] = []
    outputs = tanh(inputs)

    assert len(outputs) == 0


def test__tanh__parameters() -> None:
    tanh = Tanh()
    params = tanh.parameters()

    # Tanh has no trainable parameters
    assert len(params) == 0
    assert params == []


def test__tanh__chain_with_linear() -> None:
    # Test Tanh chained with Linear layer
    linear = Linear(2, 1, bias=False)
    tanh = Tanh()

    # Set deterministic weights
    linear.weights[0][0] = Scalar(2.0)
    linear.weights[0][1] = Scalar(-1.0)

    # Test forward pass
    inputs = [Scalar(1.0), Scalar(0.5)]
    linear_output = linear(inputs)  # 2*1 + (-1)*0.5 = 1.5
    tanh_output = tanh(linear_output)

    expected = math.tanh(1.5)
    assert tanh_output[0].data == expected

    # Test backward pass
    tanh_output[0].backward()

    # Check gradient flows correctly through the chain
    expected_tanh_grad = 1 - math.tanh(1.5) ** 2
    # d(loss)/d(w0) = tanh'(1.5) * input[0]
    assert linear.weights[0][0].grad == expected_tanh_grad * 1.0
    # d(loss)/d(w1) = tanh'(1.5) * input[1]
    assert linear.weights[0][1].grad == expected_tanh_grad * 0.5
