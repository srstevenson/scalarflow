import math
import random
from enum import Enum


class InitScheme(Enum):
    """Weight initialisation schemes."""

    HE = "he"
    GLOROT = "glorot"


def he_uniform(n_in: int) -> float:
    """He (Kaiming) uniform initialisation.

    Samples from uniform distribution U(-bound, bound) where
    bound = sqrt(6 / n_in). Designed for ReLU activations.

    Reference:
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on
        ImageNet Classification (He et al., 2015)
        <https://arxiv.org/abs/1502.01852>

    Args:
        n_in: Number of input units.

    Returns:
        Random weight value following He uniform distribution.
    """
    bound = math.sqrt(6.0 / n_in)
    return random.uniform(-bound, bound)


def glorot_uniform(n_in: int, n_out: int) -> float:
    """Glorot (Xavier) uniform initialisation.

    Samples from uniform distribution U(-bound, bound) where
    bound = sqrt(6 / (n_in + n_out)). Designed for tanh and sigmoid activations.

    Reference:
        Understanding the difficulty of training deep feedforward neural
        networks (Glorot & Bengio, 2010)
        <http://proceedings.mlr.press/v9/glorot10a.html>

    Args:
        n_in: Number of input units.
        n_out: Number of output units.

    Returns:
        Random weight value following Glorot uniform distribution.
    """
    bound = math.sqrt(6.0 / (n_in + n_out))
    return random.uniform(-bound, bound)
