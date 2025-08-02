import math

import pytest
from micrograd.scalar import Scalar


def test__scalar__init() -> None:
    scalar = Scalar(3.0)
    assert scalar.data == 3.0
    assert not scalar.op
    assert not scalar.deps


def test__scalar__repr() -> None:
    scalar = Scalar(3.0)
    assert repr(scalar) == "Scalar(data=3.0000, grad=0.0000)"


def test__scalar__pow_float() -> None:
    scalar = Scalar(2.0)
    result = scalar**3.0
    assert result.data == 8.0
    assert result.op == "^"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__pow_scalar() -> None:
    scalar1 = Scalar(2.0)
    scalar2 = Scalar(3.0)
    result = scalar1**scalar2
    assert result.data == 8.0
    assert result.op == "^"
    assert result.deps == {scalar1, scalar2}


def test__scalar__add_scalar() -> None:
    scalar1 = Scalar(1.0)
    scalar2 = Scalar(2.0)
    result = scalar1 + scalar2
    assert result.data == 3.0
    assert result.op == "+"
    assert result.deps == {scalar1, scalar2}


def test__scalar__add_float() -> None:
    scalar = Scalar(1.0)
    result = scalar + 2.0
    assert result.data == 3.0
    assert result.op == "+"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__mul_scalar() -> None:
    scalar1 = Scalar(2.0)
    scalar2 = Scalar(3.0)
    result = scalar1 * scalar2
    assert result.data == 6.0
    assert result.op == "×"
    assert result.deps == {scalar1, scalar2}


def test__scalar__mul_float() -> None:
    scalar = Scalar(2.0)
    result = scalar * 3.0
    assert result.data == 6.0
    assert result.op == "×"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__rpow_float() -> None:
    scalar = Scalar(3.0)
    result = 2.0**scalar
    assert result.data == 8.0
    assert result.op == "^"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__rpow_scalar() -> None:
    scalar1 = Scalar(2.0)
    scalar2 = Scalar(3.0)
    result = scalar1**scalar2
    assert result.data == 8.0
    assert result.op == "^"
    assert result.deps == {scalar1, scalar2}


def test__scalar__radd_float() -> None:
    scalar = Scalar(1.0)
    result = 2.0 + scalar
    assert result.data == 3.0
    assert result.op == "+"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__rmul_float() -> None:
    scalar = Scalar(3.0)
    result = 2.0 * scalar
    assert result.data == 6.0
    assert result.op == "×"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__neg() -> None:
    scalar = Scalar(2.0)
    result = -scalar
    assert result.data == -2.0
    assert result.op == "×"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__sub_scalar() -> None:
    scalar1 = Scalar(1.0)
    scalar2 = Scalar(2.0)
    result = scalar1 - scalar2
    assert result.data == -1.0
    assert result.op == "+"
    assert len(result.deps) == 2
    assert scalar1 in result.deps
    neg_scalar2 = next(dep for dep in result.deps if dep != scalar1)
    assert neg_scalar2.data == -scalar2.data
    assert neg_scalar2.op == "×"
    assert len(neg_scalar2.deps) == 2
    assert scalar2 in neg_scalar2.deps


def test__scalar__sub_float() -> None:
    scalar = Scalar(1.0)
    result = scalar - 2.0
    assert result.data == -1.0
    assert result.op == "+"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__rsub_float() -> None:
    scalar = Scalar(1.0)
    result = 2.0 - scalar
    assert result.data == 1.0
    assert result.op == "+"
    assert len(result.deps) == 2
    data_values = {dep.data for dep in result.deps}
    assert 2.0 in data_values
    assert -1.0 in data_values


def test__scalar__truediv_scalar() -> None:
    scalar1 = Scalar(3.0)
    scalar2 = Scalar(2.0)
    result = scalar1 / scalar2
    assert result.data == 1.5
    assert result.op == "×"
    assert len(result.deps) == 2
    assert scalar1 in result.deps
    inv_scalar2 = next(dep for dep in result.deps if dep != scalar1)
    assert inv_scalar2.data == 1 / scalar2.data
    assert inv_scalar2.op == "^"
    assert len(inv_scalar2.deps) == 2
    assert scalar2 in inv_scalar2.deps


def test__scalar__truediv_float() -> None:
    scalar = Scalar(3.0)
    result = scalar / 2.0
    assert result.data == 1.5
    assert result.op == "×"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__rtruediv_float() -> None:
    scalar = Scalar(2.0)
    result = 3.0 / scalar
    assert result.data == 1.5
    assert result.op == "×"
    assert len(result.deps) == 2
    # Check that one dependency is 1 / scalar
    inv_scalar = next(dep for dep in result.deps if dep.data == 0.5)
    assert inv_scalar.op == "^"
    assert scalar in inv_scalar.deps


def test__scalar__add__backward() -> None:
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x + y
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # For z = x + y: ∂z/∂x = 1, ∂z/∂y = 1
    assert x.grad == 1.0
    assert y.grad == 1.0


def test__scalar__mul__backward() -> None:
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x * y
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # For z = x * y: ∂z/∂x = y = 3.0, ∂z/∂y = x = 2.0
    assert x.grad == 3.0
    assert y.grad == 2.0


def test__scalar__pow__backward() -> None:
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x**y
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # For z = x^y: ∂z/∂x = y * x^(y-1) = 3 * 2^2 = 12
    # For z = x^y: ∂z/∂y = x^y * ln(x) = 8 * ln(2)
    assert x.grad == 12.0
    assert y.grad == 8.0 * math.log(2.0)


def test__scalar__backward_single_node() -> None:
    x = Scalar(5.0)
    x.backward()

    assert x.grad == 1.0


def test__scalar__backward_simple_chain() -> None:
    # Compute z = x + y
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x + y
    z.backward()

    # Check all gradients are computed correctly
    assert z.grad == 1.0  # Output gradient
    assert x.grad == 1.0  # ∂z/∂x = 1
    assert y.grad == 1.0  # ∂z/∂y = 1


def test__scalar__backward_complex_expression() -> None:
    # Compute z = (x * y) + (x ** 2)
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = (x * y) + (x**2)
    z.backward()

    # z = 2*3 + 2^2 = 6 + 4 = 10
    # ∂z/∂x = y + 2*x = 3 + 2*2 = 7
    # ∂z/∂y = x = 2
    assert z.grad == 1.0
    assert x.grad == 7.0
    assert y.grad == 2.0


def test__scalar__backward_nested_operations() -> None:
    # Compute z = (x + y) * (x - y)
    x = Scalar(3.0)
    y = Scalar(2.0)
    z = (x + y) * (x - y)
    z.backward()

    # z = (3+2) * (3-2) = 5 * 1 = 5
    # ∂z/∂x = (x-y) + (x+y) = 1 + 5 = 6
    # ∂z/∂y = (x-y) + (x+y) * (-1) = 1 - 5 = -4
    assert z.grad == 1.0
    assert x.grad == 6.0
    assert y.grad == -4.0


def test__scalar__backward_power_chain() -> None:
    # Compute w = (x ** y) ** z
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = Scalar(2.0)
    u = x**y  # u = 2^3 = 8
    w = u**z  # w = 8^2 = 64
    w.backward()

    # ∂w/∂u = z * u^(z-1) = 2 * 8^1 = 16
    # ∂u/∂x = y * x^(y-1) = 3 * 2^2 = 12
    # ∂u/∂y = u * ln(x) = 8 * ln(2)
    # Chain rule: ∂w/∂x = ∂w/∂u * ∂u/∂x = 16 * 12 = 192
    # Chain rule: ∂w/∂y = ∂w/∂u * ∂u/∂y = 16 * 8 * ln(2)
    # ∂w/∂z = w * ln(u) = 64 * ln(8)
    assert w.grad == 1.0
    assert x.grad == 192.0
    assert y.grad == 16.0 * 8.0 * math.log(2.0)
    assert z.grad == 64.0 * math.log(8.0)


def test__scalar__zero_grad() -> None:
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x * y + x

    z.backward()

    assert z.grad > 0.0
    assert x.grad > 0.0
    assert y.grad > 0.0

    z.zero_grad()

    assert x.grad == 0.0
    assert y.grad == 0.0
    assert z.grad == 0.0


@pytest.mark.parametrize(
    ("input_data", "expected"), [(3.0, 3.0), (-2.0, 0.0), (0.0, 0.0)]
)
def test__scalar__relu_forward(input_data: float, expected: float) -> None:
    scalar = Scalar(input_data)
    result = scalar.relu()

    assert result.data == expected
    assert result.op == "relu"
    assert result.deps == {scalar}


@pytest.mark.parametrize(
    ("input_data", "expected"), [(3.0, 1.0), (-2.0, 0.0), (0.0, 0.0)]
)
def test__scalar__relu_backward(input_data: float, expected: float) -> None:
    x = Scalar(input_data)
    y = x.relu()
    y.grad = 1.0
    y._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    assert x.grad == expected


@pytest.mark.parametrize(("input_data", "expected_grad"), [(-1.0, 0.0), (1.0, 2.0)])
def test__scalar__relu_chain(input_data: float, expected_grad: float) -> None:
    x = Scalar(input_data)
    y = x.relu() * 2.0
    y.backward()

    assert x.grad == expected_grad
