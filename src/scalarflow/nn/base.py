from abc import ABC, abstractmethod
from typing import Any

from scalarflow import Scalar


class Module(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # pyright: ignore[reportExplicitAny, reportAny]  # noqa: ANN401

    @abstractmethod
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module."""
        ...
