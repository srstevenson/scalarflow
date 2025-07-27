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
    assert result.deps == {scalar}


def test__scalar__pow_scalar() -> None:
    scalar1 = Scalar(2.0)
    scalar2 = Scalar(3.0)
    result = scalar1**scalar2
    assert result.data == 8.0
    assert result.op == "^"
    assert result.deps == {scalar1, scalar2}


def test__scalar__rpow() -> None:
    scalar = Scalar(3.0)
    result = 2.0**scalar
    assert result.data == 8.0
    assert result.op == "^"
    assert result.deps == {scalar}


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
    assert result.deps == {scalar}


def test__scalar__radd_float() -> None:
    scalar = Scalar(1.0)
    result = 2.0 + scalar
    assert result.data == 3.0
    assert result.op == "+"
    assert result.deps == {scalar}


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
    assert result.deps == {scalar}


def test__scalar__rmul_float() -> None:
    scalar = Scalar(3.0)
    result = 2.0 * scalar
    assert result.data == 6.0
    assert result.op == "×"
    assert result.deps == {scalar}


def test__scalar__neg() -> None:
    scalar = Scalar(2.0)
    result = -scalar
    assert result.data == -2.0
    assert result.op == "×"
    assert result.deps == {scalar}


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
    assert neg_scalar2.deps == {scalar2}


def test__scalar__sub_float() -> None:
    scalar = Scalar(1.0)
    result = scalar - 2.0
    assert result.data == -1.0
    assert result.op == "+"
    assert result.deps == {scalar}


def test__scalar__rsub_float() -> None:
    scalar = Scalar(1.0)
    result = 2.0 - scalar
    assert result.data == 1.0
    assert result.op == "+"
    assert len(result.deps) == 1
    neg_scalar = next(iter(result.deps))
    assert neg_scalar.data == -scalar.data
    assert neg_scalar.op == "×"
    assert neg_scalar.deps == {scalar}


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
    assert inv_scalar2.deps == {scalar2}


def test__scalar__truediv_float() -> None:
    scalar = Scalar(3.0)
    result = scalar / 2.0
    assert result.data == 1.5
    assert result.op == "×"
    assert result.deps == {scalar}


def test__scalar__rtruediv_float() -> None:
    scalar = Scalar(2.0)
    result = 3.0 / scalar
    assert result.data == 1.5
    assert result.op == "×"
    assert len(result.deps) == 1
    inv_scalar = next(iter(result.deps))
    assert inv_scalar.data == 1 / scalar.data
    assert inv_scalar.op == "^"
    assert inv_scalar.deps == {scalar}
