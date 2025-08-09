import math

from scalarflow.nn.init import glorot_uniform, he_uniform


def test__he_uniform__bounds() -> None:
    n_in = 100
    expected_bound = math.sqrt(6.0 / n_in)

    # Test many samples to ensure they're within bounds
    for _ in range(1000):
        value = he_uniform(n_in)
        assert -expected_bound <= value <= expected_bound


def test__he_uniform__different_n_in() -> None:
    # Test that different n_in values give different bounds
    n_in_small = 10
    n_in_large = 1000

    bound_small = math.sqrt(6.0 / n_in_small)
    bound_large = math.sqrt(6.0 / n_in_large)

    # Smaller n_in should give larger bounds
    assert bound_small > bound_large

    # Test samples are within their respective bounds
    value_small = he_uniform(n_in_small)
    value_large = he_uniform(n_in_large)

    assert -bound_small <= value_small <= bound_small
    assert -bound_large <= value_large <= bound_large


def test__he_uniform__distribution_coverage() -> None:
    n_in = 64
    bound = math.sqrt(6.0 / n_in)

    # Generate many samples and check distribution properties
    samples = [he_uniform(n_in) for _ in range(10000)]

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
    n_in = 100
    n_out = 50
    expected_bound = math.sqrt(6.0 / (n_in + n_out))

    # Test many samples to ensure they're within bounds
    for _ in range(1000):
        value = glorot_uniform(n_in, n_out)
        assert -expected_bound <= value <= expected_bound


def test__glorot_uniform__different_n_values() -> None:
    # Test that different n_in/n_out values give different bounds
    n_in_small, n_out_small = 10, 5
    n_in_large, n_out_large = 1000, 500

    bound_small = math.sqrt(6.0 / (n_in_small + n_out_small))
    bound_large = math.sqrt(6.0 / (n_in_large + n_out_large))

    # Smaller n values should give larger bounds
    assert bound_small > bound_large

    # Test samples are within their respective bounds
    value_small = glorot_uniform(n_in_small, n_out_small)
    value_large = glorot_uniform(n_in_large, n_out_large)

    assert -bound_small <= value_small <= bound_small
    assert -bound_large <= value_large <= bound_large


def test__glorot_uniform__distribution_coverage() -> None:
    n_in = 64
    n_out = 32
    bound = math.sqrt(6.0 / (n_in + n_out))

    # Generate many samples and check distribution properties
    samples = [glorot_uniform(n_in, n_out) for _ in range(10000)]

    # Check that we get values across the range
    min_val = min(samples)
    max_val = max(samples)

    # Should cover most of the range (allowing for some randomness)
    assert min_val < -bound * 0.8
    assert max_val > bound * 0.8

    # Mean should be close to zero
    mean = sum(samples) / len(samples)
    assert abs(mean) < 0.05
