from abc import ABC, abstractmethod

from scalarflow import Scalar


class Module(ABC):
    @abstractmethod
    def __call__(self, inputs: list[Scalar]) -> list[Scalar]: ...

    @abstractmethod
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module."""
        ...
