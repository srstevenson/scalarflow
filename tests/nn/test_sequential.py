import pytest

from scalarflow import Scalar
from scalarflow.nn import Linear, ReLU, Sequential, Tanh


def test__sequential__init() -> None:
    linear = Linear(3, 2)
    relu = ReLU()
    model = Sequential([linear, relu])

    assert len(model.modules) == 2
    assert model.modules[0] is linear
    assert model.modules[1] is relu


def test__sequential__init_with_empty_list_raises_error() -> None:
    with pytest.raises(ValueError, match="Sequential requires at least one module"):
        Sequential([])


def test__sequential__forward_pass() -> None:
    linear1 = Linear(2, 3, bias=False)
    relu = ReLU()
    linear2 = Linear(3, 1, bias=False)

    linear1.weights = [
        [Scalar(1.0), Scalar(2.0)],
        [Scalar(3.0), Scalar(4.0)],
        [Scalar(5.0), Scalar(6.0)],
    ]
    linear2.weights = [[Scalar(0.5), Scalar(-0.5), Scalar(1.0)]]

    model = Sequential([linear1, relu, linear2])

    inputs = [Scalar(1.0), Scalar(2.0)]

    # linear1 gives [1*1+2*2, 3*1+4*2, 5*1+6*2] = [5, 11, 17]
    # relu gives [5, 11, 17]
    # linear2 gives 5*0.5 + 11*(-0.5) + 17*1.0 = 2.5 - 5.5 + 17 = 14.0
    expected_output = 14.0

    output = model(inputs)
    assert len(output) == 1
    assert output[0].data == expected_output


def test__sequential__parameters() -> None:
    linear1 = Linear(2, 2)
    relu = ReLU()
    linear2 = Linear(2, 1)
    model = Sequential([linear1, relu, linear2])

    expected_params = linear1.parameters() + linear2.parameters()
    actual_params = model.parameters()

    assert len(actual_params) == len(expected_params)
    assert all(a is b for a, b in zip(actual_params, expected_params, strict=True))


def test__sequential__parameters_with_no_trainable_modules() -> None:
    model = Sequential([ReLU(), Tanh()])
    assert model.parameters() == []


def test__sequential__gradient_flow() -> None:
    linear = Linear(1, 1, bias=True)
    linear.weights[0][0] = Scalar(2.0)
    assert linear.biases is not None
    linear.biases[0] = Scalar(1.0)

    model = Sequential([linear])
    input_scalar = Scalar(3.0)

    output = model([input_scalar])[0]
    assert output.data == 7.0

    output.backward()

    assert linear.weights[0][0].grad == 3.0
    assert linear.biases[0].grad == 1.0
    assert input_scalar.grad == 2.0
