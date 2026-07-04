"""Generic supervised-learning training loop for PyTorch models."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.optim import Optimizer

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainerConfig:
    """Configuration options for Trainer.

    Attributes:
        epochs: Number of training epochs to run.
        device: Torch device string to run training on (e.g. "cpu", "cuda").
        checkpoint_dir: Directory to save checkpoints to. If empty,
            checkpoints are never saved.
        checkpoint_every: Save a checkpoint every N epochs (1 = every epoch).
    """

    epochs: int = 1
    device: str = "cpu"
    checkpoint_dir: str = ""
    checkpoint_every: int = 1


class Trainer:
    """Full training pipeline for a supervised PyTorch model.

    Wraps the train/validate loop, checkpointing, and logging so
    individual experiment scripts only need to assemble a model,
    optimizer, loss function, and dataloaders.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: LossFn,
        config: TrainerConfig | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Model to train; moved to `config.device`.
            optimizer: Optimizer bound to `model`'s parameters.
            loss_fn: Callable computing the loss from (predictions, targets).
            config: Trainer options; defaults to `TrainerConfig()`.
        """
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        train_loader: Iterable[Batch],
        val_loader: Iterable[Batch] | None = None,
    ) -> dict[str, list[float]]:
        """Run the training loop for `config.epochs` epochs.

        Args:
            train_loader: Batches of (inputs, targets) used for training.
            val_loader: Optional batches of (inputs, targets) used for
                per-epoch validation.

        Returns:
            The training history: a mapping of "train_loss"/"val_loss" to
            a list of per-epoch values.
        """
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            self.history["train_loss"].append(train_loss)
            message = (
                f"Epoch {epoch}/{self.config.epochs} - train_loss: {train_loss:.4f}"
            )

            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, train=False)
                self.history["val_loss"].append(val_loss)
                message += f" - val_loss: {val_loss:.4f}"

            logger.info(message)

            if self.config.checkpoint_dir and epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch)

        return self.history

    def _run_epoch(self, loader: Iterable[Batch], train: bool) -> float:
        """Run a single train or evaluation pass over `loader`.

        Args:
            loader: Batches of (inputs, targets).
            train: If True, run in training mode with gradient updates;
                otherwise run in evaluation mode under `torch.no_grad()`.

        Returns:
            The average loss over all batches.
        """
        self.model.train(mode=train)

        total_loss = 0.0
        n_batches = 0

        with torch.enable_grad() if train else torch.no_grad():
            for batch_inputs, batch_targets in loader:
                inputs = batch_inputs.to(self.device)
                targets = batch_targets.to(self.device)

                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()  # type: ignore[no-untyped-call]
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, epoch: int) -> Path:
        """Save model and optimizer state to `config.checkpoint_dir`.

        Args:
            epoch: Current epoch number, used to name the checkpoint file.

        Returns:
            The path the checkpoint was written to.
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        logger.debug(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Restore model and optimizer state from a checkpoint file.

        Args:
            checkpoint_path: Path to a checkpoint file written by
                `save_checkpoint`.

        Returns:
            The epoch number the checkpoint was saved at.
        """
        checkpoint: dict[str, Any] = torch.load(
            Path(checkpoint_path), map_location=self.device
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return int(checkpoint["epoch"])
