from functools import partial
from typing import override

from scalarflow import Scalar
from scalarflow.nn.base import Module
from scalarflow.nn.init import InitScheme, glorot_uniform, he_uniform


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
                init_fn = partial(he_uniform, fan_in=in_features)
            case InitScheme.GLOROT:
                init_fn = partial(
                    glorot_uniform, fan_in=in_features, fan_out=out_features
                )

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
            raise ValueError(f"Expected {self.in_features} inputs, got {len(inputs)}")

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
