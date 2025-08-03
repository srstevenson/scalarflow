import math

from scalarflow import Scalar
from scalarflow.nn import Linear, Sigmoid


def test__sigmoid__forward_pass() -> None:
    sigmoid = Sigmoid()

    inputs = [Scalar(-2.0), Scalar(-1.0), Scalar(0.0), Scalar(1.0), Scalar(2.0)]
    outputs = sigmoid(inputs)

    assert len(outputs) == len(inputs)
    # Test against expected sigmoid values: 1 / (1 + exp(-x))
    assert outputs[0].data == 1 / (1 + math.exp(2.0))
    assert outputs[1].data == 1 / (1 + math.exp(1.0))
    assert outputs[2].data == 0.5  # sigmoid(0) = 0.5
    assert outputs[3].data == 1 / (1 + math.exp(-1.0))
    assert outputs[4].data == 1 / (1 + math.exp(-2.0))


def test__sigmoid__empty_input() -> None:
    sigmoid = Sigmoid()

    inputs: list[Scalar] = []
    outputs = sigmoid(inputs)

    assert len(outputs) == 0


def test__sigmoid__parameters() -> None:
    sigmoid = Sigmoid()
    params = sigmoid.parameters()

    # Sigmoid has no trainable parameters
    assert len(params) == 0
    assert params == []


def test__sigmoid__chain_with_linear() -> None:
    # Test Sigmoid chained with Linear layer
    linear = Linear(2, 1, bias=False)
    sigmoid = Sigmoid()

    # Set deterministic weights
    linear.weights[0][0] = Scalar(2.0)
    linear.weights[0][1] = Scalar(-1.0)

    # Test forward pass
    inputs = [Scalar(1.0), Scalar(0.5)]
    linear_output = linear(inputs)  # 2*1 + (-1)*0.5 = 1.5
    sigmoid_output = sigmoid(linear_output)

    expected = 1 / (1 + math.exp(-1.5))
    assert sigmoid_output[0].data == expected

    # Test backward pass
    sigmoid_output[0].backward()

    # Check gradient flows correctly through the chain
    # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    sigmoid_val = 1 / (1 + math.exp(-1.5))
    expected_sigmoid_grad = sigmoid_val * (1 - sigmoid_val)
    # d(loss)/d(w0) = sigmoid'(1.5) * input[0]
    assert linear.weights[0][0].grad == expected_sigmoid_grad * 1.0
    # d(loss)/d(w1) = sigmoid'(1.5) * input[1]
    assert linear.weights[0][1].grad == expected_sigmoid_grad * 0.5
