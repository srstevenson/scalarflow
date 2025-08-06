from abc import ABC, abstractmethod

from scalarflow import Scalar


class Module(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> list[Scalar]: ...  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]  # noqa: ANN002, ANN003

    @abstractmethod
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module."""
        ...
