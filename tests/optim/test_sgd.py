import pytest

from scalarflow import Scalar
from scalarflow.optim import SGD


def test__sgd__init() -> None:
    params = [Scalar(1.0), Scalar(2.0), Scalar(3.0)]
    lr = 0.01

    sgd = SGD(params, lr)

    assert sgd.params == params
    assert sgd.lr == lr


@pytest.mark.parametrize("lr", [0.0, -0.01, -1.0])
def test__sgd__init_invalid_lr(lr: float) -> None:
    with pytest.raises(ValueError, match=f"Learning rate must be positive, got {lr}"):
        SGD([], lr)


def test__sgd__step_multiple_params() -> None:
    param1 = Scalar(10.0)
    param1.grad = 1.0

    param2 = Scalar(-5.0)
    param2.grad = -2.0

    param3 = Scalar(0.0)
    param3.grad = 0.5

    sgd = SGD([param1, param2, param3], 0.2)
    sgd.step()

    assert param1.data == 9.8
    assert param2.data == -4.6
    assert param3.data == -0.1


def test__sgd__zero_grad_multiple_params() -> None:
    param1 = Scalar(1.0)
    param1.grad = 1.5

    param2 = Scalar(2.0)
    param2.grad = -2.5

    param3 = Scalar(3.0)
    param3.grad = 0.0

    sgd = SGD([param1, param2, param3], 0.01)
    sgd.zero_grad()

    assert param1.data == 1.0
    assert param2.data == 2.0
    assert param3.data == 3.0

    assert param1.grad == 0.0
    assert param2.grad == 0.0
    assert param3.grad == 0.0


def test__sgd__multiple_steps() -> None:
    param = Scalar(10.0)
    lr = 0.5

    sgd = SGD([param], lr)

    # First step
    param.grad = 2.0
    sgd.step()
    assert param.data == 9.0

    # Second step with different gradient
    param.grad = -1.0
    sgd.step()
    assert param.data == 9.5

    # Third step
    param.grad = 1.0
    sgd.step()
    assert param.data == 9.0


def test__sgd__step_and_zero_grad_workflow() -> None:
    param = Scalar(1.0)
    param.grad = 0.8

    sgd = SGD([param], 0.25)

    # Perform step
    sgd.step()
    assert param.data == 0.8
    assert param.grad == 0.8

    # Zero gradients
    sgd.zero_grad()
    assert param.data == 0.8
    assert param.grad == 0.0
