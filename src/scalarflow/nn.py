from abc import ABC, abstractmethod
from typing import Any


class Module(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # pyright: ignore[reportExplicitAny, reportAny]  # noqa: ANN401
