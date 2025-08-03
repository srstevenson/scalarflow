import math

import pytest

from scalarflow import Scalar
from scalarflow.nn import InitScheme, Linear, ReLU, Tanh, glorot_uniform, he_uniform


def test__he_uniform__bounds() -> None:
    fan_in = 100
    expected_bound = math.sqrt(6.0 / fan_in)

    # Test many samples to ensure they're within bounds
    for _ in range(1000):
        value = he_uniform(fan_in)
        assert -expected_bound <= value <= expected_bound


def test__he_uniform__different_fan_in() -> None:
    # Test that different fan_in values give different bounds
    fan_in_small = 10
    fan_in_large = 1000

    bound_small = math.sqrt(6.0 / fan_in_small)
    bound_large = math.sqrt(6.0 / fan_in_large)

    # Smaller fan_in should give larger bounds
    assert bound_small > bound_large

    # Test samples are within their respective bounds
    value_small = he_uniform(fan_in_small)
    value_large = he_uniform(fan_in_large)

    assert -bound_small <= value_small <= bound_small
    assert -bound_large <= value_large <= bound_large


def test__he_uniform__distribution_coverage() -> None:
    fan_in = 64
    bound = math.sqrt(6.0 / fan_in)

    # Generate many samples and check distribution properties
    samples = [he_uniform(fan_in) for _ in range(10000)]

    # Check that we get values across the range
    min_val = min(samples)
    max_val = max(samples)

    # Should cover most of the range (allowing for some randomness)
    assert min_val < -bound * 0.8
    assert max_val > bound * 0.8

    # Mean should be close to zero
    mean = sum(samples) / len(samples)
    assert abs(mean) < 0.05


def test__glorot_uniform__bounds() -> None:
    fan_in = 100
    fan_out = 50
    expected_bound = math.sqrt(6.0 / (fan_in + fan_out))

    # Test many samples to ensure they're within bounds
    for _ in range(1000):
        value = glorot_uniform(fan_in, fan_out)
        assert -expected_bound <= value <= expected_bound


def test__glorot_uniform__different_fan_values() -> None:
    # Test that different fan_in/fan_out values give different bounds
    fan_in_small, fan_out_small = 10, 5
    fan_in_large, fan_out_large = 1000, 500

    bound_small = math.sqrt(6.0 / (fan_in_small + fan_out_small))
    bound_large = math.sqrt(6.0 / (fan_in_large + fan_out_large))

    # Smaller fan values should give larger bounds
    assert bound_small > bound_large

    # Test samples are within their respective bounds
    value_small = glorot_uniform(fan_in_small, fan_out_small)
    value_large = glorot_uniform(fan_in_large, fan_out_large)

    assert -bound_small <= value_small <= bound_small
    assert -bound_large <= value_large <= bound_large


def test__glorot_uniform__distribution_coverage() -> None:
    fan_in = 64
    fan_out = 32
    bound = math.sqrt(6.0 / (fan_in + fan_out))

    # Generate many samples and check distribution properties
    samples = [glorot_uniform(fan_in, fan_out) for _ in range(10000)]

    # Check that we get values across the range
    min_val = min(samples)
    max_val = max(samples)

    # Should cover most of the range (allowing for some randomness)
    assert min_val < -bound * 0.8
    assert max_val > bound * 0.8

    # Mean should be close to zero
    mean = sum(samples) / len(samples)
    assert abs(mean) < 0.05


def test__linear__init() -> None:
    linear = Linear(3, 2)

    assert linear.in_features == 3
    assert linear.out_features == 2
    assert linear.use_bias is True
    assert len(linear.weights) == 2  # out_features rows
    assert len(linear.weights[0]) == 3  # in_features columns
    assert linear.biases is not None
    assert len(linear.biases) == 2  # out_features biases


@pytest.mark.parametrize("init_scheme", [InitScheme.HE, InitScheme.GLOROT])
def test__linear__init_schemes(init_scheme: InitScheme) -> None:
    linear = Linear(3, 2, init=init_scheme)

    assert linear.in_features == 3
    assert linear.out_features == 2
    assert len(linear.weights) == 2
    assert len(linear.weights[0]) == 3
    assert linear.biases is not None
    assert len(linear.biases) == 2


def test__linear__init_without_bias() -> None:
    linear = Linear(3, 2, bias=False)

    assert linear.in_features == 3
    assert linear.out_features == 2
    assert linear.use_bias is False
    assert linear.biases is None


def test__linear__call_with_bias() -> None:
    linear = Linear(2, 1, bias=True)

    # Set deterministic weights and bias for testing
    linear.weights[0][0] = Scalar(1.0)
    linear.weights[0][1] = Scalar(2.0)
    assert linear.biases is not None
    linear.biases[0] = Scalar(0.5)

    inputs = [Scalar(3.0), Scalar(4.0)]
    outputs = linear(inputs)

    # 1*3 + 2*4 + 0.5 = 11.5
    assert len(outputs) == 1
    assert outputs[0].data == 11.5


def test__linear__call_without_bias() -> None:
    linear = Linear(2, 1, bias=False)

    # Set deterministic weights for testing
    linear.weights[0][0] = Scalar(1.0)
    linear.weights[0][1] = Scalar(2.0)

    inputs = [Scalar(3.0), Scalar(4.0)]
    outputs = linear(inputs)

    # 1*3 + 2*4 = 11.0
    assert len(outputs) == 1
    assert outputs[0].data == 11.0


def test__linear__call_multiple_outputs() -> None:
    linear = Linear(2, 2, bias=False)

    # Set deterministic weights
    linear.weights[0][0] = Scalar(1.0)
    linear.weights[0][1] = Scalar(2.0)
    linear.weights[1][0] = Scalar(3.0)
    linear.weights[1][1] = Scalar(4.0)

    inputs = [Scalar(1.0), Scalar(2.0)]
    outputs = linear(inputs)

    # [1*1 + 2*2, 3*1 + 4*2] = [5.0, 11.0]
    assert len(outputs) == 2
    assert outputs[0].data == 5.0
    assert outputs[1].data == 11.0


def test__linear__call_input_validation() -> None:
    linear = Linear(3, 2)

    with pytest.raises(ValueError, match="Expected 3 inputs, got 2"):
        linear([Scalar(1.0), Scalar(2.0)])

    with pytest.raises(ValueError, match="Expected 3 inputs, got 4"):
        linear([Scalar(1.0), Scalar(2.0), Scalar(3.0), Scalar(4.0)])


def test__linear__parameters_with_bias() -> None:
    linear = Linear(2, 2, bias=True)
    params = linear.parameters()

    # 2*2 weights + 2 biases
    assert len(params) == 6


def test__linear__parameters_without_bias() -> None:
    linear = Linear(2, 2, bias=False)
    params = linear.parameters()

    # 2*2 weights
    assert len(params) == 4


def test__linear__gradient_flow() -> None:
    linear = Linear(1, 1, bias=True)

    # Set simple weights
    linear.weights[0][0] = Scalar(2.0)
    assert linear.biases is not None
    linear.biases[0] = Scalar(1.0)

    # Forward pass
    input_scalar = Scalar(3.0)
    output = linear([input_scalar])[0]

    # Expected output: 2*3 + 1 = 7
    assert output.data == 7.0

    # Backward pass
    output.backward()

    # Check gradients
    assert linear.weights[0][0].grad == 3.0  # d(output)/d(weight) = input
    assert linear.biases is not None
    assert linear.biases[0].grad == 1.0  # d(output)/d(bias) = 1
    assert input_scalar.grad == 2.0  # d(output)/d(input) = weight


def test__linear__zero_grad_integration() -> None:
    linear = Linear(1, 1)

    # Run forward and backward to accumulate gradients
    input_scalar = Scalar(1.0)
    output = linear([input_scalar])[0]
    output.backward()

    # Zero gradients
    for param in linear.parameters():
        param.zero_grad()
    input_scalar.zero_grad()

    # Check all gradients are zero
    for param in linear.parameters():
        assert param.grad == 0.0
    assert input_scalar.grad == 0.0


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


def test__relu__single_input() -> None:
    relu = ReLU()

    # Test positive input
    positive_input = [Scalar(3.0)]
    positive_output = relu(positive_input)
    assert len(positive_output) == 1
    assert positive_output[0].data == 3.0

    # Test negative input
    negative_input = [Scalar(-1.5)]
    negative_output = relu(negative_input)
    assert len(negative_output) == 1
    assert negative_output[0].data == 0.0


def test__relu__parameters() -> None:
    relu = ReLU()
    params = relu.parameters()

    # ReLU has no trainable parameters
    assert len(params) == 0
    assert params == []


def test__relu__gradient_flow() -> None:
    relu = ReLU()

    # Test positive input (gradient should flow through)
    positive_input = Scalar(2.0)
    positive_output = relu([positive_input])[0]
    positive_output.backward()

    assert positive_input.grad == 1.0  # Gradient flows through

    # Reset gradients
    positive_input.zero_grad()

    # Test negative input (gradient should be blocked)
    negative_input = Scalar(-1.0)
    negative_output = relu([negative_input])[0]
    negative_output.backward()

    assert negative_input.grad == 0.0  # Gradient is blocked


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


def test__relu__multiple_applications() -> None:
    relu = ReLU()

    # Test that applying ReLU multiple times is idempotent for positive values
    inputs = [Scalar(5.0), Scalar(-3.0)]
    first_pass = relu(inputs)
    second_pass = relu(first_pass)

    assert first_pass[0].data == second_pass[0].data == 5.0
    assert first_pass[1].data == second_pass[1].data == 0.0


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
    assert abs(tanh_output[0].data - expected) < 1e-10

    # Test backward pass
    tanh_output[0].backward()

    # Check gradient flows correctly through the chain
    expected_tanh_grad = 1 - math.tanh(1.5) ** 2
    assert (
        abs(linear.weights[0][0].grad - expected_tanh_grad * 1.0) < 1e-10
    )  # d(loss)/d(w0) = tanh'(1.5) * input[0]
    assert (
        abs(linear.weights[0][1].grad - expected_tanh_grad * 0.5) < 1e-10
    )  # d(loss)/d(w1) = tanh'(1.5) * input[1]
