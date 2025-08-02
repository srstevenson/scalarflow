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


@pytest.mark.parametrize("other", [Scalar(3.0), 3.0])
def test__scalar__pow(other: Scalar | float) -> None:
    scalar = Scalar(2.0)
    result = scalar**other

    assert result.data == 8.0
    assert result.op == "^"
    assert len(result.deps) == 2
    assert scalar in result.deps


@pytest.mark.parametrize("other", [Scalar(2.0), 2.0])
def test__scalar__add(other: Scalar | float) -> None:
    scalar = Scalar(1.0)
    result = scalar + other

    assert result.data == 3.0
    assert result.op == "+"
    assert len(result.deps) == 2
    assert scalar in result.deps


@pytest.mark.parametrize("other", [Scalar(3.0), 3.0])
def test__scalar__mul(other: Scalar | float) -> None:
    scalar = Scalar(2.0)
    result = scalar * other

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


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [(1.0, math.tanh(1.0)), (-1.0, math.tanh(-1.0)), (0.0, 0.0)],
)
def test__scalar__tanh_forward(input_data: float, expected: float) -> None:
    scalar = Scalar(input_data)
    result = scalar.tanh()

    assert result.data == expected
    assert result.op == "tanh"
    assert result.deps == {scalar}


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [(1.0, 1 - math.tanh(1.0) ** 2), (-1.0, 1 - math.tanh(-1.0) ** 2), (0.0, 1.0)],
)
def test__scalar__tanh_backward(input_data: float, expected: float) -> None:
    x = Scalar(input_data)
    y = x.tanh()
    y.grad = 1.0
    y._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    assert x.grad == expected


@pytest.mark.parametrize(
    ("input_data", "expected_grad"),
    [(1.0, 2.0 * (1 - math.tanh(1.0) ** 2)), (-1.0, 2.0 * (1 - math.tanh(-1.0) ** 2))],
)
def test__scalar__tanh_chain(input_data: float, expected_grad: float) -> None:
    x = Scalar(input_data)
    y = x.tanh() * 2.0
    y.backward()

    assert x.grad == expected_grad


@pytest.mark.parametrize("input_data", [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
def test__scalar__sigmoid_forward(input_data: float) -> None:
    scalar = Scalar(input_data)
    result = scalar.sigmoid()

    expected = 1 / (1 + math.exp(-input_data))
    assert result.data == expected
    assert result.op == "sigmoid"
    assert result.deps == {scalar}


def _grad_sigmoid(x: float) -> float:
    sigmoid_val = 1 / (1 + math.exp(-x))
    return sigmoid_val * (1 - sigmoid_val)


@pytest.mark.parametrize("input_data", [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
def test__scalar__sigmoid_backward(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.sigmoid()
    y.grad = 1.0
    y._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    assert x.grad == _grad_sigmoid(x.data)


@pytest.mark.parametrize("input_data", [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
def test__scalar__sigmoid_chain(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.sigmoid() * 2.0
    y.backward()

    expected_grad = 2.0 * _grad_sigmoid(input_data)
    assert x.grad == expected_grad


@pytest.mark.parametrize("input_data", [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
def test__scalar__exp_forward(input_data: float) -> None:
    scalar = Scalar(input_data)
    result = scalar.exp()

    expected = math.exp(input_data)
    assert result.data == expected
    assert result.op == "exp"
    assert result.deps == {scalar}


@pytest.mark.parametrize("input_data", [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
def test__scalar__exp_backward(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.exp()
    y.grad = 1.0
    y._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    expected_grad = math.exp(input_data)
    assert x.grad == expected_grad


@pytest.mark.parametrize("input_data", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test__scalar__exp_chain(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.exp() * 2.0
    y.backward()

    expected_grad = 2.0 * math.exp(input_data)
    assert x.grad == expected_grad


@pytest.mark.parametrize("input_data", [0.1, 0.5, 1.0, 2.0, 3.0, math.e])
def test__scalar__log_forward(input_data: float) -> None:
    scalar = Scalar(input_data)
    result = scalar.log()

    expected = math.log(input_data)
    assert result.data == expected
    assert result.op == "log"
    assert result.deps == {scalar}


@pytest.mark.parametrize("input_data", [0.1, 0.5, 1.0, 2.0, 3.0, math.e])
def test__scalar__log_backward(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.log()
    y.grad = 1.0
    y._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    expected_grad = 1 / input_data
    assert x.grad == expected_grad


@pytest.mark.parametrize("input_data", [0.1, 0.5, 1.0, 2.0, 3.0])
def test__scalar__log_chain(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.log() * 2.0
    y.backward()

    expected_grad = 2.0 * (1 / input_data)
    assert x.grad == expected_grad


@pytest.mark.parametrize("input_data", [0.25, 1.0, 4.0, 9.0, 16.0, 25.0])
def test__scalar__sqrt_forward(input_data: float) -> None:
    scalar = Scalar(input_data)
    result = scalar.sqrt()

    expected = math.sqrt(input_data)
    assert result.data == expected
    assert result.op == "sqrt"
    assert result.deps == {scalar}


@pytest.mark.parametrize("input_data", [0.25, 1.0, 4.0, 9.0, 16.0, 25.0])
def test__scalar__sqrt_backward(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.sqrt()
    y.grad = 1.0
    y._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    expected_grad = 1 / (2 * math.sqrt(input_data))
    assert x.grad == expected_grad


@pytest.mark.parametrize("input_data", [0.25, 1.0, 4.0, 9.0, 16.0])
def test__scalar__sqrt_chain(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.sqrt() * 2.0
    y.backward()

    expected_grad = 2.0 * (1 / (2 * math.sqrt(input_data)))
    assert x.grad == expected_grad


@pytest.mark.parametrize("input_data", [-2.0, 0.0, 2.0])
def test__scalar__abs_forward(input_data: float) -> None:
    scalar = Scalar(input_data)
    result = scalar.abs()

    expected = abs(input_data)
    assert result.data == expected
    assert result.op == "abs"
    assert result.deps == {scalar}


@pytest.mark.parametrize("input_data", [-2.0, 0.0, 2.0])
def test__scalar__abs_backward(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.abs()
    y.grad = 1.0
    y._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    if input_data > 0:
        expected_grad = 1
    elif input_data < 0:
        expected_grad = -1
    else:  # input_data == 0, following PyTorch convention
        expected_grad = 0
    assert x.grad == expected_grad


@pytest.mark.parametrize("input_data", [-2.0, 0.0, 2.0])
def test__scalar__abs_chain(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.abs() * 2.0
    y.backward()

    if input_data > 0:
        expected_grad = 2.0
    elif input_data < 0:
        expected_grad = -2.0
    else:
        expected_grad = 0.0
    assert x.grad == expected_grad


@pytest.mark.parametrize("other", [Scalar(3.0), 3.0])
def test__scalar__min_forward_self_smaller(other: Scalar | float) -> None:
    scalar = Scalar(2.0)
    result = scalar.min(other)

    assert result.data == 2.0
    assert result.op == "min"
    assert len(result.deps) == 2
    assert scalar in result.deps


@pytest.mark.parametrize("other", [Scalar(1.0), 1.0])
def test__scalar__min_forward_other_smaller(other: Scalar | float) -> None:
    scalar = Scalar(2.0)
    result = scalar.min(other)

    assert result.data == 1.0
    assert result.op == "min"
    assert len(result.deps) == 2
    assert scalar in result.deps


@pytest.mark.parametrize("other", [Scalar(2.0), 2.0])
def test__scalar__min_forward_equal(other: Scalar | float) -> None:
    scalar = Scalar(2.0)
    result = scalar.min(other)

    assert result.data == 2.0
    assert result.op == "min"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__min_backward_self_smaller() -> None:
    x = Scalar(1.0)
    y = Scalar(2.0)
    z = x.min(y)
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # x is smaller, so gradient flows to x
    assert x.grad == 1.0
    assert y.grad == 0.0


def test__scalar__min_backward_other_smaller() -> None:
    x = Scalar(2.0)
    y = Scalar(1.0)
    z = x.min(y)
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # y is smaller, so gradient flows to y
    assert x.grad == 0.0
    assert y.grad == 1.0


def test__scalar__min_backward_equal() -> None:
    x = Scalar(2.0)
    y = Scalar(2.0)
    z = x.min(y)
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # Equal values, so gradient is split equally
    assert x.grad == 0.5
    assert y.grad == 0.5


def test__scalar__min_chain() -> None:
    x = Scalar(1.0)
    y = Scalar(3.0)
    z = x.min(y) * 2.0
    z.backward()

    # x is smaller, so gradient flows to x and is multiplied by 2
    assert x.grad == 2.0
    assert y.grad == 0.0


@pytest.mark.parametrize("other", [Scalar(1.0), 1.0])
def test__scalar__max_forward_self_larger(other: Scalar | float) -> None:
    scalar = Scalar(2.0)
    result = scalar.max(other)

    assert result.data == 2.0
    assert result.op == "max"
    assert len(result.deps) == 2
    assert scalar in result.deps


@pytest.mark.parametrize("other", [Scalar(3.0), 3.0])
def test__scalar__max_forward_other_larger(other: Scalar | float) -> None:
    scalar = Scalar(2.0)
    result = scalar.max(other)

    assert result.data == 3.0
    assert result.op == "max"
    assert len(result.deps) == 2
    assert scalar in result.deps


@pytest.mark.parametrize("other", [Scalar(2.0), 2.0])
def test__scalar__max_forward_equal(other: Scalar | float) -> None:
    scalar = Scalar(2.0)
    result = scalar.max(other)

    assert result.data == 2.0
    assert result.op == "max"
    assert len(result.deps) == 2
    assert scalar in result.deps


def test__scalar__max_backward_self_larger() -> None:
    x = Scalar(3.0)
    y = Scalar(1.0)
    z = x.max(y)
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # x is larger, so gradient flows to x.
    assert x.grad == 1.0
    assert y.grad == 0.0


def test__scalar__max_backward_other_larger() -> None:
    x = Scalar(1.0)
    y = Scalar(3.0)
    z = x.max(y)
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # y is larger, so gradient flows to y.
    assert x.grad == 0.0
    assert y.grad == 1.0


def test__scalar__max_backward_equal() -> None:
    x = Scalar(2.0)
    y = Scalar(2.0)
    z = x.max(y)
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # Equal values, so gradient is split equally.
    assert x.grad == 0.5
    assert y.grad == 0.5


def test__scalar__max_chain() -> None:
    x = Scalar(3.0)
    y = Scalar(1.0)
    z = x.max(y) * 2.0
    z.backward()

    # x is larger, so gradient flows to x and is multiplied by 2.
    assert x.grad == 2.0
    assert y.grad == 0.0


@pytest.mark.parametrize("input_data", [0.0, math.pi / 4, math.pi / 2, math.pi])
def test__scalar__sin_forward(input_data: float) -> None:
    scalar = Scalar(input_data)
    result = scalar.sin()

    expected = math.sin(input_data)
    assert result.data == expected
    assert result.op == "sin"
    assert result.deps == {scalar}


@pytest.mark.parametrize("input_data", [0.0, math.pi / 4, math.pi / 2, math.pi])
def test__scalar__sin_backward(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.sin()
    y.grad = 1.0
    y._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    expected_grad = math.cos(input_data)
    assert x.grad == expected_grad


@pytest.mark.parametrize("input_data", [0.0, math.pi / 4, math.pi / 2])
def test__scalar__sin_chain(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.sin() * 2.0
    y.backward()

    assert x.grad == 2.0 * math.cos(input_data)


@pytest.mark.parametrize("input_data", [0.0, math.pi / 4, math.pi / 2, math.pi])
def test__scalar__cos_forward(input_data: float) -> None:
    scalar = Scalar(input_data)
    result = scalar.cos()

    assert result.data == math.cos(input_data)
    assert result.op == "cos"
    assert result.deps == {scalar}


@pytest.mark.parametrize("input_data", [0.0, math.pi / 4, math.pi / 2, math.pi])
def test__scalar__cos_backward(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.cos()
    y.grad = 1.0
    y._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    assert x.grad == -math.sin(input_data)


@pytest.mark.parametrize("input_data", [0.0, math.pi / 4, math.pi / 2])
def test__scalar__cos_chain(input_data: float) -> None:
    x = Scalar(input_data)
    y = x.cos() * 2.0
    y.backward()

    assert x.grad == 2.0 * -math.sin(input_data)


@pytest.mark.parametrize("input_data", [-1.0, 0.5, 2.0])
def test__scalar__clamp_both_bounds(input_data: float) -> None:
    scalar = Scalar(input_data)
    result = scalar.clamp(0.0, 1.0)

    expected = max(0.0, min(1.0, input_data))
    assert result.data == expected


def test__scalar__clamp_single_bounds() -> None:
    # Min only.
    x = Scalar(-1.0)
    result = x.clamp(min_val=0.0)
    assert result.data == 0.0

    # Max only.
    x = Scalar(2.0)
    result = x.clamp(max_val=1.0)
    assert result.data == 1.0

    # No bounds.
    x = Scalar(0.5)
    result = x.clamp()
    assert result.data == 0.5
    assert result is x


def test__scalar__clamp_backward() -> None:
    x = Scalar(0.5)
    y = x.clamp(0.0, 1.0)
    y.backward()

    # Value within bounds, gradient should flow through
    assert x.grad == 1.0


def test__scalar__clamp_chain() -> None:
    x = Scalar(0.5)
    y = x.clamp(0.0, 1.0) * 2.0
    y.backward()

    assert x.grad == 2.0


def test__scalar__clamp_invalid_bounds() -> None:
    x = Scalar(0.5)

    with pytest.raises(ValueError, match=r"min_val \(1.0\) must be <= max_val \(0.0\)"):
        x.clamp(1.0, 0.0)
