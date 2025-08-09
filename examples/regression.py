import math
import random
from dataclasses import dataclass

from scalarflow import Scalar, nn, optim

type Dataset = list[tuple[float, float]]


@dataclass(frozen=True)
class Params:
    """Parameters for the training process.

    Attributes:
        seed: The random seed to use.
        n_samples: The number of samples to generate.
        lr: The learning rate.
        n_epochs: The number of epochs to train for.
    """

    seed: int = 42
    n_samples: int = 50
    lr: float = 0.001
    n_epochs: int = 1000


def generate_data(*, n_samples: int) -> Dataset:
    """Generate a dataset of quadratic data.

    Args:
        n_samples: The number of samples to generate.

    Returns:
        A list of tuples, where each tuple contains an x and y value.
    """
    xs = [random.uniform(-2, 2) for _ in range(n_samples)]
    return [(x, x**2) for x in xs]


def train_sample(model: nn.Module, loss_fn: nn.Module, x: float, y: float) -> float:
    """Train the model on a single sample.

    Args:
        model: The model to train.
        loss_fn: The loss function to use.
        x: The input value.
        y: The target value.

    Returns:
        The loss for the sample.
    """
    y_hat = model([Scalar(x)])
    loss = loss_fn(y_hat, [Scalar(y)])[0]
    loss.backward()
    return loss.data


def train_epoch(
    model: nn.Module, loss_fn: nn.Module, dataset: Dataset, optimiser: optim.Optimiser
) -> float:
    """Train the model for one epoch.

    Args:
        model: The model to train.
        loss_fn: The loss function to use.
        dataset: The dataset to train on.
        optimiser: The optimiser to use for parameter updates.

    Returns:
        The total loss for the epoch.
    """
    total_loss = 0.0
    for x, y in dataset:
        total_loss += train_sample(model, loss_fn, x, y)

    optimiser.step()
    optimiser.zero_grad()

    return total_loss


def evaluate_epoch(model: nn.Module, loss_fn: nn.Module, dataset: Dataset) -> float:
    """Evaluate the model for one epoch.

    Args:
        model: The model to evaluate.
        loss_fn: The loss function to use.
        dataset: The dataset to evaluate on.

    Returns:
        The total loss for the epoch.
    """
    total_loss = 0.0
    for x, y in dataset:
        y_hat = model([Scalar(x)])
        loss = loss_fn(y_hat, [Scalar(y)])[0]
        total_loss += loss.data
    return total_loss


def main() -> None:
    """Train a model to learn a quadratic function."""
    params = Params()

    random.seed(params.seed)

    dataset = generate_data(n_samples=params.n_samples)
    split_index = int(len(dataset) * 0.75)
    train_dataset = dataset[:split_index]
    val_dataset = dataset[split_index:]

    model = nn.Sequential([nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1)])
    loss_fn = nn.MSELoss()
    optimiser = optim.SGD(model.parameters(), params.lr)

    print("Epoch | Train Loss | Train RMSE | Val Loss | Val RMSE")
    print("----- | ---------- | ---------- | -------- | --------")

    for epoch in range(params.n_epochs + 1):
        train_loss = train_epoch(model, loss_fn, train_dataset, optimiser)
        val_loss = evaluate_epoch(model, loss_fn, val_dataset)

        if epoch % 100 == 0:
            avg_train_loss = train_loss / len(train_dataset)
            avg_val_loss = val_loss / len(val_dataset)
            train_rmse = math.sqrt(avg_train_loss)
            val_rmse = math.sqrt(avg_val_loss)
            print(
                f"{epoch:>5} | {avg_train_loss:>10.4f} | {train_rmse:>10.4f} | "
                f"{avg_val_loss:>8.4f} | {val_rmse:>8.4f}"
            )


if __name__ == "__main__":
    main()
