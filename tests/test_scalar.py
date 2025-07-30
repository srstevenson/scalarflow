import math

from micrograd.scalar import Scalar


def test__scalar__init() -> None:
    scalar = Scalar(3.0)
    assert scalar.data == 3.0
    assert not scalar.op
    assert not scalar.deps


def test__scalar__repr() -> None:
    scalar = Scalar(3.0)
    assert repr(scalar) == "Scalar(data=3.0)"


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


def test__scalar__add_backward() -> None:
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x + y
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # For z = x + y: ∂z/∂x = 1, ∂z/∂y = 1
    assert x.grad == 1.0
    assert y.grad == 1.0


def test__scalar__mul_backward() -> None:
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x * y
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # For z = x * y: ∂z/∂x = y = 3.0, ∂z/∂y = x = 2.0
    assert x.grad == 3.0
    assert y.grad == 2.0


def test__scalar__pow_backward() -> None:
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x**y
    z.grad = 1.0
    z._backward()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # For z = x^y: ∂z/∂x = y * x^(y-1) = 3 * 2^2 = 12
    # For z = x^y: ∂z/∂y = x^y * ln(x) = 8 * ln(2)
    assert x.grad == 12.0
    assert y.grad == 8.0 * math.log(2.0)
