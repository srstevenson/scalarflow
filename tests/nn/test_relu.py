from scalarflow import Scalar
from scalarflow.nn import Linear, ReLU


def test__relu__forward_pass() -> None:
    relu = ReLU()

    inputs = [Scalar(-2.0), Scalar(-0.5), Scalar(0.0), Scalar(0.5), Scalar(2.0)]
    outputs = relu(inputs)

    assert len(outputs) == len(inputs)
    assert outputs[0].data == 0.0  # max(0, -2.0)
    assert outputs[1].data == 0.0  # max(0, -0.5)
    assert outputs[2].data == 0.0  # max(0, 0.0)
    assert outputs[3].data == 0.5  # max(0, 0.5)
    assert outputs[4].data == 2.0  # max(0, 2.0)


def test__relu__empty_input() -> None:
    relu = ReLU()

    inputs: list[Scalar] = []
    outputs = relu(inputs)

    assert len(outputs) == 0


def test__relu__parameters() -> None:
    relu = ReLU()
    params = relu.parameters()

    assert params == []


def test__relu__chain_with_linear() -> None:
    # Test ReLU chained with Linear layer
    linear = Linear(2, 1, bias=False)
    relu = ReLU()

    # Set deterministic weights
    linear.weights[0][0] = Scalar(1.0)
    linear.weights[0][1] = Scalar(-1.0)

    # Test case where linear output is positive
    inputs_positive = [Scalar(2.0), Scalar(1.0)]
    linear_output = linear(inputs_positive)  # 1*2 + (-1)*1 = 1.0
    relu_output = relu(linear_output)

    assert relu_output[0].data == 1.0  # ReLU(1.0) = 1.0

    # Test case where linear output is negative
    inputs_negative = [Scalar(1.0), Scalar(2.0)]
    linear_output = linear(inputs_negative)  # 1*1 + (-1)*2 = -1.0
    relu_output = relu(linear_output)

    assert relu_output[0].data == 0.0  # ReLU(-1.0) = 0.0
