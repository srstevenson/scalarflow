import math
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, override

from scalarflow import Scalar


class InitScheme(Enum):
    """Weight initialisation schemes."""

    HE = "he"
    GLOROT = "glorot"


def he_uniform(fan_in: int) -> float:
    """He (Kaiming) uniform initialization.

    Samples from uniform distribution U(-bound, bound) where
    bound = sqrt(6 / fan_in). Designed for ReLU-family activations.

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
    """Glorot (Xavier) uniform initialization.

    Samples from uniform distribution U(-bound, bound) where
    bound = sqrt(6 / (fan_in + fan_out)). Designed for tanh/sigmoid activations.

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


class Module(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # pyright: ignore[reportExplicitAny, reportAny]  # noqa: ANN401

    @abstractmethod
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module."""
        ...


class Linear(Module):
    """A linear (fully connected) layer.

    Applies a linear transformation to incoming data: y = xW^T + b where W is
    the weight matrix and b is the bias vector.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        init: InitScheme = InitScheme.HE,
    ) -> None:
        """Initialise the linear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias to the output.
            init: Weight initialisation scheme to use.
        """
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.use_bias: bool = bias

        match init:
            case InitScheme.HE:
                init_fn = lambda: he_uniform(in_features)  # noqa: E731
            case InitScheme.GLOROT:
                init_fn = lambda: glorot_uniform(in_features, out_features)  # noqa: E731

        self.weights: list[list[Scalar]] = [
            [Scalar(init_fn()) for _ in range(in_features)] for _ in range(out_features)
        ]
        self.biases: list[Scalar] | None = (
            [Scalar(0.0) for _ in range(out_features)] if bias else None
        )

    @override
    def __call__(self, inputs: list[Scalar]) -> list[Scalar]:
        """Forward pass through the linear layer.

        Args:
            inputs: List of input scalars, length must equal in_features.

        Returns:
            List of output scalars, length equals out_features.

        Raises:
            ValueError: If input length doesn't match in_features.
        """
        if len(inputs) != self.in_features:
            msg = f"Expected {self.in_features} inputs, got {len(inputs)}"
            raise ValueError(msg)

        outputs = [
            sum((w * i for w, i in zip(weights, inputs, strict=True)), Scalar(0.0))
            for weights in self.weights
        ]

        if self.biases is not None:
            outputs = [o + b for o, b in zip(outputs, self.biases, strict=True)]

        return outputs

    @override
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the layer.

        Returns:
            List containing all weight and bias scalars.
        """
        params: list[Scalar] = []

        for weights in self.weights:
            params.extend(weights)

        if self.biases is not None:
            params.extend(self.biases)

        return params


class ReLU(Module):
    """ReLU activation function module.

    Applies the Rectified Linear Unit function element-wise to input scalars.
    ReLU(x) = max(0, x).
    """

    @override
    def __call__(self, inputs: list[Scalar]) -> list[Scalar]:
        """Apply ReLU activation to input scalars.

        Args:
            inputs: List of input scalars.

        Returns:
            List of output scalars with ReLU activation applied.
        """
        return [scalar.relu() for scalar in inputs]

    @override
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module.

        Returns:
            Empty list as ReLU has no trainable parameters.
        """
        return []


class Tanh(Module):
    """Tanh activation function module.

    Applies the hyperbolic tangent function element-wise to input scalars.
    Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)).
    """

    @override
    def __call__(self, inputs: list[Scalar]) -> list[Scalar]:
        """Apply Tanh activation to input scalars.

        Args:
            inputs: List of input scalars.

        Returns:
            List of output scalars with Tanh activation applied.
        """
        return [scalar.tanh() for scalar in inputs]

    @override
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module.

        Returns:
            Empty list as Tanh has no trainable parameters.
        """
        return []
