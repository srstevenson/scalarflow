import math
import random
from dataclasses import dataclass

from scalarflow import Scalar, nn, optim

type Dataset = list[tuple[tuple[float, float], int]]


@dataclass(frozen=True)
class Params:
    """Parameters for the training process.

    Attributes:
        seed: The random seed to use.
        n_samples: Total number of samples to generate.
        lr: The learning rate.
        n_epochs: The number of epochs to train for.
    """

    seed: int = 42
    n_samples: int = 800
    lr: float = 0.01
    n_epochs: int = 50


def generate_dataset(*, n_samples: int) -> Dataset:
    """Generate a non-linearly separable binary dataset of concentric circles.

    Returns a balanced dataset of two classes placed on noisy concentric
    circles. The negative class is near radius 0.7 and the positive class near
    radius 1.4.
    """
    half = n_samples // 2
    data: Dataset = []

    for _ in range(half):
        radius = random.gauss(0.7, 0.08)
        theta = random.uniform(0.0, 2 * math.pi)
        data.append(((radius * math.cos(theta), radius * math.sin(theta)), 0))

    for _ in range(n_samples - half):
        radius = random.gauss(1.4, 0.10)
        theta = random.uniform(0.0, 2 * math.pi)
        data.append(((radius * math.cos(theta), radius * math.sin(theta)), 1))

    random.shuffle(data)
    return data


def train_sample(
    model: nn.Module, loss_fn: nn.Module, sample: tuple[float, float], target: int
) -> float:
    """Train the model on a single sample and return the loss value."""
    x1, x2 = sample
    y_hat = model([Scalar(x1), Scalar(x2)])
    loss = loss_fn(y_hat, [Scalar(target)])[0]
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
    for sample, label in dataset:
        total_loss += train_sample(model, loss_fn, sample, label)
    optimiser.step()
    optimiser.zero_grad()
    return total_loss


def evaluate_epoch(
    model: nn.Module, loss_fn: nn.Module, dataset: Dataset
) -> tuple[float, float]:
    """Evaluate the model for one epoch.

    Args:
        model: The model to evaluate.
        loss_fn: The loss function to use.
        dataset: The dataset to evaluate on.

    Returns:
        A tuple containing the total loss and accuracy for the epoch.
    """
    total_loss = 0.0
    correct = 0
    for sample, target in dataset:
        x1, x2 = sample
        y_hat = model([Scalar(x1), Scalar(x2)])
        loss = loss_fn(y_hat, [Scalar(target)])[0]
        total_loss += loss.data
        pred = int(y_hat[0].data >= 0.5)  # noqa: PLR2004
        if pred == target:
            correct += 1
    accuracy = correct / len(dataset)
    return total_loss, accuracy


def main() -> None:
    """Train a small network to classify two concentric circles."""
    params = Params()

    random.seed(params.seed)

    dataset = generate_dataset(n_samples=params.n_samples)
    split_index = int(len(dataset) * 0.75)
    train_dataset = dataset[:split_index]
    val_dataset = dataset[split_index:]

    model = nn.Sequential([nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()])
    loss_fn = nn.BCELoss()
    optimiser = optim.SGD(model.parameters(), params.lr)

    print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
    print("----- | ---------- | --------- | -------- | -------")

    for epoch in range(params.n_epochs + 1):
        train_loss = train_epoch(model, loss_fn, train_dataset, optimiser)

        if epoch % 10 == 0:
            train_loss, train_acc = evaluate_epoch(model, loss_fn, train_dataset)
            mean_train_loss = train_loss / len(train_dataset)
            val_loss, val_acc = evaluate_epoch(model, loss_fn, val_dataset)
            mean_val_loss = val_loss / len(val_dataset)
            print(
                f"{epoch:>5} | {mean_train_loss:>10.4f} | {train_acc:>9.3f} | "
                f"{mean_val_loss:>8.4f} | {val_acc:>7.3f}"
            )


if __name__ == "__main__":
    main()
