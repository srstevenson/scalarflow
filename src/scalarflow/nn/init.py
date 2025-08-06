import math
import random
from enum import Enum


class InitScheme(Enum):
    """Weight initialisation schemes."""

    HE = "he"
    GLOROT = "glorot"


def he_uniform(fan_in: int) -> float:
    """He (Kaiming) uniform initialisation.

    Samples from uniform distribution U(-bound, bound) where
    bound = √(6 / fan_in). Designed for ReLU-family activations.

    Reference:
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on
        ImageNet Classification (He et al., 2015)
        https://arxiv.org/abs/1502.01852

    Args:
        fan_in: Number of input units.

    Returns:
        Random weight value following He uniform distribution.
    """
    bound = math.sqrt(6.0 / fan_in)
    return random.uniform(-bound, bound)


def glorot_uniform(fan_in: int, fan_out: int) -> float:
    """Glorot (Xavier) uniform initialisation.

    Samples from uniform distribution U(-bound, bound) where
    bound = √(6 / (fan_in + fan_out)). Designed for tanh/sigmoid activations.

    Reference:
        Understanding the difficulty of training deep feedforward neural
        networks (Glorot & Bengio, 2010)
        http://proceedings.mlr.press/v9/glorot10a.html

    Args:
        fan_in: Number of input units.
        fan_out: Number of output units.

    Returns:
        Random weight value following Glorot uniform distribution.
    """
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(-bound, bound)
