from abc import ABC, abstractmethod

from scalarflow import Scalar


class Module(ABC):
    """Abstract base class for neural network modules.

    This class defines the interface for all neural network components in
    ScalarFlow. Concrete module implementations should inherit from this
    class and implement the required abstract methods.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> list[Scalar]:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]  # noqa: ANN002, ANN003
        """Forward pass through the module.

        This method should implement the forward computation of the module,
        transforming input scalars into output scalars according to the
        module's functionality.

        Args:
            *args: Variable positional arguments (typically input scalars).
            **kwargs: Variable keyword arguments.

        Returns:
            List of output scalars produced by the module.
        """
        ...

    @abstractmethod
    def parameters(self) -> list[Scalar]:
        """Return all trainable parameters in the module.

        This method should return all scalar parameters that should be
        updated during training. For modules without trainable parameters
        (like activation functions), this should return an empty list.

        Returns:
            List of all trainable scalar parameters in the module.
        """
        ...
