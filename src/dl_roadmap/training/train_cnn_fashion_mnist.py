"""Training entry point for the CNN Fashion-MNIST classifier."""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dl_roadmap.engine import ClassPredictor, Trainer, TrainerConfig
from dl_roadmap.metrics import evaluate_multiclass_classification
from dl_roadmap.models import CnnFashionMNIST
from dl_roadmap.utils import (
    LoadConfig,
    LoggerConfig,
    load_model,
    save_model,
    seed_everything,
    setup_logger,
)
from dl_roadmap.visualization import plot_confusion_matrix, plot_training_history


def _generate_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Load Fashion-MNIST (downloading it if needed) into train/val DataLoaders.

    Args:
        config: Data section of the experiment config with keys: root, batch_size.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    root = config.get("root", "data/raw")
    batch_size = config.get("batch_size", 64)

    logger.info(f"Loading Fashion-MNIST from '{root}'")

    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    val_dataset = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )

    logger.info(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def _run_training(
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: dict[str, Any],
    skip_training: bool,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Build the model, optimizer, and Trainer, then run the training loop.

    Args:
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        training_config: Training section of the experiment config with keys:
            epochs, device, checkpoint_dir, checkpoint_every, learning_rate,
            model_path.
        skip_training: If True, load weights from `training_config.model_path`
            instead of training, and return an empty history.

    Returns:
        Tuple of (trained model, training history dict with "train_loss" and
        "val_loss" per-epoch lists; empty if `skip_training` is True).
    """
    model_path = training_config.get("model_path", "model/cnn_fashion_mnist.pt")
    model = CnnFashionMNIST()
    logger.info(f"Model: {type(model).__name__}")

    if skip_training:
        load_model(model, model_path)
        return model, {}

    trainer_config = TrainerConfig(
        epochs=training_config.get("epochs", 1),
        device=training_config.get("device"),
        checkpoint_dir=training_config.get("checkpoint_dir", ""),
        checkpoint_every=training_config.get("checkpoint_every", 1),
        show_progress=training_config.get("show_progress", True),
    )

    lr = training_config.get("learning_rate", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    logger.info(f"Optimizer: Adam(lr={lr}), Loss: CrossEntropyLoss")

    trainer = Trainer(model, optimizer, loss_fn, trainer_config)
    history = trainer.fit(train_loader, val_loader)

    save_model(trainer.model, model_path)

    return trainer.model, history


def _save_plots(
    model: nn.Module,
    val_loader: DataLoader,
    history: dict[str, list[float]],
    visualization_config: dict[str, Any],
) -> None:
    """Save the loss curve, a validation-set confusion matrix, and log metrics.

    Args:
        model: Trained model, evaluated on `val_loader` for the confusion
            matrix and validation metrics.
        val_loader: DataLoader for the validation split.
        history: Training history dict with "train_loss" and "val_loss" lists.
        visualization_config: Visualization section of the experiment config with keys:
            figures_dir, show_fig.
    """
    figures_dir = Path(visualization_config.get("figures_dir", "reports/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    show_fig = visualization_config.get("show_fig", False)

    if history:
        loss_path = figures_dir / "train-val-loss.png"
        plot_training_history(**history, filename=str(loss_path), show_fig=show_fig)
        logger.info(f"Saved training plot: {loss_path}")

    predictor = ClassPredictor(model)
    X_val = torch.cat([images for images, _ in val_loader])
    y_val = torch.cat([labels for _, labels in val_loader])
    y_pred = predictor.predict(X_val)

    confusion_path = figures_dir / "confusion-matrix.png"
    plot_confusion_matrix(
        y_pred, y_val, filename=str(confusion_path), show_fig=show_fig
    )
    logger.info(f"Saved confusion matrix: {confusion_path}")

    _log_statistics(model, y_pred, y_val)


def _log_statistics(
    model: nn.Module, y_pred: torch.Tensor, y_val: torch.Tensor
) -> None:
    """Log validation metrics and parameter count for the model.

    Args:
        model: Model to report the parameter count for.
        y_pred: Predicted class labels for the validation set.
        y_val: True class labels for the validation set.
    """
    metrics = evaluate_multiclass_classification(y_pred, y_val)
    parameters = sum(p.numel() for p in model.parameters())

    stats = pd.DataFrame([{**metrics, "parameters": parameters}], index=["CNN"])
    logger.info(f"Validation statistics:\n{stats.to_string()}")


def main(
    config_file: str = "configs/03_cnn_fashion_mnist.yaml",
    skip_training: bool = False,
) -> None:
    """Load config, configure logging, and run training.

    Args:
        config_file: Path to the experiment YAML config.
        skip_training: If True, load a previously saved model instead of
            training (see `training.model_path` in the config).
    """
    config_loader = LoadConfig(config_file)
    config = config_loader.config

    setup_logger(LoggerConfig(**config.get("logging", {})))

    seed_everything(config.get("seed", None))

    train_loader, val_loader = _generate_dataloaders(config.get("data", {}))

    model, history = _run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=config.get("training", {}),
        skip_training=skip_training,
    )

    _save_plots(model, val_loader, history, config.get("visualization", {}))


if __name__ == "__main__":
    typer.run(main)
