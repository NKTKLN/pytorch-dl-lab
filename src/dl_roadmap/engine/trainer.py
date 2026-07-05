"""Generic supervised-learning training loop for PyTorch models."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainerConfig:
    """Configuration options for Trainer.

    Attributes:
        epochs: Number of training epochs to run.
        device: Torch device string (e.g. "cpu", "cuda"). None auto-selects
            CUDA if available, otherwise CPU.
        checkpoint_dir: Directory to save checkpoints to. If empty,
            checkpoints are never saved.
        checkpoint_every: Save a checkpoint every N epochs (1 = every epoch).
        show_progress: Whether to display a tqdm progress bar during training.
    """

    epochs: int = 1
    device: torch.device | str | None = None
    checkpoint_dir: str = ""
    checkpoint_every: int = 1
    show_progress: bool = True


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

        self.device = self.config.device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        logger.debug(
            f"Trainer initialized: model={type(model).__name__}, "
            f"optimizer={type(optimizer).__name__}, device={self.device}"
        )

    @staticmethod
    def _loader_len(loader: Iterable[Batch]) -> int | None:
        """Return loader length if available, otherwise None."""
        try:
            return len(loader)  # type: ignore[arg-type]
        except TypeError:
            return None

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
        logger.debug(
            f"Starting training: epochs={self.config.epochs}, "
            f"val={'yes' if val_loader is not None else 'no'}, "
            f"checkpoint_dir='{self.config.checkpoint_dir or 'disabled'}'"
        )

        batches_per_epoch = self._loader_len(train_loader)
        total_steps = batches_per_epoch and self.config.epochs * batches_per_epoch
        epoch_width = len(str(self.config.epochs))

        bar_format = (
            "{desc}{percentage:3.0f}%[{bar:20}] {n_fmt}/{total_fmt} "
            ":: eta={remaining}{postfix}"
        )
        pbar = tqdm(
            total=total_steps,
            desc=f"Epoch {1:>{epoch_width}}/{self.config.epochs}",
            ascii=" >=",
            bar_format=bar_format,
            leave=True,
            disable=not self.config.show_progress,
        )
        last_val_loss: float | None = None

        for epoch in range(1, self.config.epochs + 1):
            pbar.set_description(f"Epoch {epoch:>{epoch_width}}/{self.config.epochs}")

            train_loss = self._run_epoch(
                train_loader,
                train=True,
                pbar=pbar,
                last_val_loss=last_val_loss,
            )
            self.history["train_loss"].append(train_loss)

            loss_data = {"train_loss": f"{train_loss:.4g}"}

            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, train=False)
                self.history["val_loss"].append(val_loss)
                last_val_loss = val_loss
                loss_data["val_loss"] = f"{val_loss:.4g}"

            pbar.set_postfix(**loss_data)

            if self.config.checkpoint_dir and epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch)

        pbar.close()
        logger.debug("Training complete")
        return self.history

    def _run_epoch(
        self,
        loader: Iterable[Batch],
        train: bool,
        pbar: Any | None = None,
        last_val_loss: float | None = None,
    ) -> float:
        """Run a single train or evaluation pass over `loader`.

        Args:
            loader: Batches of (inputs, targets).
            train: If True, run in training mode with gradient updates;
                otherwise run in evaluation mode under `torch.no_grad()`.
            pbar: Progress bar to update with running loss after each batch.
                If None, no progress bar is updated.
            last_val_loss: Validation loss from the previous epoch, shown in
                the progress bar postfix alongside the running train loss.
                None hides it (e.g. before the first validation pass).

        Returns:
            The average loss over all batches.
        """
        mode = "train" if train else "eval"
        logger.debug(f"Running epoch in {mode} mode")
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

                if pbar is not None:
                    running_loss = total_loss / n_batches

                    postfix = {"train_loss": f"{running_loss:.4g}"}
                    if last_val_loss is not None:
                        postfix["val_loss"] = f"{last_val_loss:.4g}"

                    pbar.set_postfix(**postfix)
                    pbar.update(1)

        avg_loss = total_loss / max(n_batches, 1)
        logger.debug(f"Epoch {mode} pass: batches={n_batches}, avg_loss={avg_loss:.4f}")
        return avg_loss

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
            checkpoint_path: Path to a checkpoint file written by `save_checkpoint`.

        Returns:
            The epoch number the checkpoint was saved at.
        """
        checkpoint: dict[str, Any] = torch.load(
            Path(checkpoint_path), map_location=self.device
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = int(checkpoint["epoch"])
        logger.debug(f"Loaded checkpoint: {checkpoint_path}, epoch={epoch}")
        return epoch
