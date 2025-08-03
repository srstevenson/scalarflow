import math

from scalarflow.nn.init import glorot_uniform, he_uniform


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
