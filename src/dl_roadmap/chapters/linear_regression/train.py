"""Training entry point for the linear regression example."""

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_roadmap.data import make_synthetic_regression_dataset
from dl_roadmap.engine import Trainer, TrainerConfig
from dl_roadmap.utils import LoadConfig, LoggerConfig, seed_everything, setup_logger
from dl_roadmap.visualization import plot_training_history


def _generate_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Generate synthetic regression data and wrap it into train/val DataLoaders.

    Args:
        config: Data section of the experiment config with keys:
            n_samples, n_features, noise, batch_size.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    n_features = config.get("n_features", 1)
    n_samples = config.get("n_samples", 250)
    noise = config.get("noise", 25.0)
    batch_size = config.get("batch_size", 32)

    logger.info(
        f"Generating dataset: n_samples={n_samples}, "
        f"n_features={n_features}, noise={noise}"
    )

    X_train, X_val, y_train, y_val, real_coef, real_bias = (
        make_synthetic_regression_dataset(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
        )
    )

    logger.info(f"Dataset split: train={len(X_train)}, val={len(X_val)}")
    logger.info(f"Real params: coef={real_coef}, bias={real_bias:.4f}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def _run_training(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_features: int,
    training_config: dict[str, Any],
) -> dict[str, list[float]]:
    """Build the model, optimizer, and Trainer, then run the training loop.

    Args:
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        n_features: Number of input features; determines the model input size.
        training_config: Training section of the experiment config with keys:
            epochs, device, checkpoint_dir, checkpoint_every, learning_rate.

    Returns:
        Training history dict with "train_loss" and "val_loss" per-epoch lists.
    """
    trainer_config = TrainerConfig(
        epochs=training_config.get("epochs", 1),
        device=training_config.get("device"),
        checkpoint_dir=training_config.get("checkpoint_dir", ""),
        checkpoint_every=training_config.get("checkpoint_every", 1),
    )

    model = nn.Linear(n_features, 1)
    logger.info(f"Model: Linear({n_features} -> 1)")

    lr = training_config.get("learning_rate", 1e-2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    logger.info(f"Optimizer: SGD(lr={lr}), Loss: MSELoss")

    trainer = Trainer(model, optimizer, loss_fn, trainer_config)
    history = trainer.fit(train_loader, val_loader)

    logger.info(
        f"Model params: coef={model.weight.detach().numpy()}, "
        f"bias={model.bias.detach().item()}"
    )
    return history


def _save_plots(
    history: dict[str, list[float]],
    visualization_config: dict[str, Any],
) -> None:
    """Save the train/val loss plot to the configured figures directory.

    Args:
        history: Training history dict with "train_loss" and "val_loss" lists.
        visualization_config: Visualization section of the experiment config with keys:
            figures_dir, show_fig.
    """
    figures_dir = Path(visualization_config.get("figures_dir", "reports/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "train-val-loss.png"

    plot_training_history(
        **history,
        filename=str(figure_path),
        show_fig=visualization_config.get("show_fig", False),
    )
    logger.info(f"Saved training plot: {figure_path}")


def main() -> None:
    """Load config, configure logging, and run training."""
    config_loader = LoadConfig("configs/01_linear_regression.yaml")
    config = config_loader.config

    setup_logger(LoggerConfig(**config.get("logging", {})))

    seed_everything(config.get("seed", None))

    data_config = config.get("data", {})
    train_loader, val_loader = _generate_dataloaders(data_config)

    history = _run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        n_features=data_config.get("n_features", 1),
        training_config=config.get("training", {}),
    )

    _save_plots(history, config.get("visualization", {}))


if __name__ == "__main__":
    main()
