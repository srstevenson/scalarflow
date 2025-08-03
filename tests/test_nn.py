import math

import pytest

from scalarflow import Scalar
from scalarflow.nn import Linear, glorot_uniform, he_uniform


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
