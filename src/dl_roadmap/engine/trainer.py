"""Generic supervised-learning training loop for PyTorch models."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from tqdm import tqdm

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
EpochCallback = Callable[[int, float, float | None], None]


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
        patience: Stop training early after this many consecutive epochs
            without a val_loss improvement greater than `min_delta`. 0
            disables early stopping. Requires a `val_loader`.
        min_delta: Minimum decrease in val_loss (vs. the best seen so far)
            to count as an improvement for early stopping.
        restore_best_weights: If True, reload the model weights from the
            best epoch (lowest val_loss) once training ends.
        show_progress: Whether to display a tqdm progress bar during training.
        grad_clip_norm: Max gradient norm for `clip_grad_norm_`, applied
            after `backward()` and before `optimizer.step()`. None disables
            gradient clipping.
    """

    epochs: int = 1
    device: torch.device | str | None = None
    checkpoint_dir: str = ""
    checkpoint_every: int = 1
    patience: int = 0
    min_delta: float = 0.0
    restore_best_weights: bool = False
    show_progress: bool = True
    grad_clip_norm: float | None = None


class Trainer:
    """Full training pipeline for a supervised PyTorch model.

    Wraps the train/validate loop, checkpointing, and logging so
    individual experiment scripts only need to assemble a model,
    optimizer, loss function, and dataloaders. Subclasses that need the
    model's forward call to see more than just the inputs (e.g. teacher
    forcing) should override `_forward` rather than `_run_epoch`.

    Attributes:
        history: Per-epoch "train_loss"/"val_loss" values, populated by `fit`.
        best_epoch: The 1-indexed epoch with the lowest val_loss seen so far,
            set by `fit` when `config.patience > 0`. None until then.
        callbacks: Callables invoked once per epoch, after the LR scheduler
            step and before early-stopping bookkeeping.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: LossFn,
        scheduler: LRScheduler | None = None,
        config: TrainerConfig | None = None,
        callbacks: list[EpochCallback] | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Model to train; moved to `config.device`.
            optimizer: Optimizer bound to `model`'s parameters.
            loss_fn: Callable computing the loss from (predictions, targets).
            scheduler: Optional LR scheduler stepped once per epoch. If it is
                a `ReduceLROnPlateau`, `fit` must be called with a `val_loader`.
            config: Trainer options; defaults to `TrainerConfig()`.
            callbacks: Callables of the form `(epoch, train_loss, val_loss)`,
                each invoked once at the end of every epoch (`val_loss` is
                None when `fit` is called without a `val_loader`). Useful for
                per-epoch side effects the trainer doesn't know about itself,
                e.g. decaying a model's teacher-forcing ratio.
        """
        self.config = config or TrainerConfig()

        self.device = self.config.device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.callbacks = callbacks or []

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_epoch: int | None = None

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

    def _step_scheduler(self, val_loss: float | None = None) -> None:
        """Advance the LR scheduler by one step.

        Args:
            val_loss: Current epoch's validation loss. Required when the
                scheduler is a `ReduceLROnPlateau`, which steps on this
                metric instead of unconditionally each epoch.

        Raises:
            ValueError: If the scheduler is a `ReduceLROnPlateau` but no
                `val_loss` was provided (i.e. `fit` was called without a
                `val_loader`).
        """
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if val_loss is None:
                raise ValueError(
                    "ReduceLROnPlateau requires a validation loss; "
                    "call fit() with a val_loader."
                )

            self.scheduler.step(float(val_loss))
        else:
            self.scheduler.step()

    def _forward(
        self, inputs: torch.Tensor, _targets: torch.Tensor, _train: bool
    ) -> torch.Tensor:
        """Computes model predictions for a batch.

        Override to pass extra context into the model's forward call, e.g.
        ground-truth targets for teacher forcing in a seq2seq decoder.
        Ignored here; the base implementation only needs `inputs`.

        Args:
            inputs: Batch inputs, already moved to `self.device`.
            _targets: Batch targets, already moved to `self.device`.
            _train: Whether this call happens during a training pass.

        Returns:
            Model predictions, passed to `self.loss_fn` alongside targets.
        """
        return self.model(inputs)  # type: ignore[no-any-return]

    def fit(
        self,
        train_loader: Iterable[Batch],
        val_loader: Iterable[Batch] | None = None,
    ) -> None:
        """Run the training loop for `config.epochs` epochs.

        Args:
            train_loader: Batches of (inputs, targets) used for training.
            val_loader: Optional batches of (inputs, targets) used for
                per-epoch validation. Required if `config.patience > 0`.

        Raises:
            ValueError: If `config.patience > 0` but no `val_loader` is given,
                or if `scheduler` is a `ReduceLROnPlateau` and no `val_loader`
                is given.
        """
        if self.config.patience > 0 and val_loader is None:
            raise ValueError("Early stopping (patience > 0) requires a val_loader.")

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
            desc=f"epoch {1:>{epoch_width}}/{self.config.epochs}",
            ascii=" >=",
            bar_format=bar_format,
            leave=True,
            disable=not self.config.show_progress,
        )

        best_val_loss = float("inf")
        last_val_loss: float | None = None
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            pbar.set_description(f"epoch {epoch:>{epoch_width}}/{self.config.epochs}")

            train_loss = self._run_epoch(
                train_loader,
                train=True,
                pbar=pbar,
                last_val_loss=last_val_loss,
            )
            self.history["train_loss"].append(train_loss)

            loss_data = {"train_loss": f"{train_loss:.4g}"}

            val_loss: float | None = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, train=False)
                self.history["val_loss"].append(val_loss)
                last_val_loss = val_loss
                loss_data["val_loss"] = f"{val_loss:.4g}"
                loss_data["loss_gap"] = f"{abs(train_loss - val_loss):.4g}"  # TODO

            self._step_scheduler(val_loss)

            for callback in self.callbacks:
                callback(epoch, train_loss, val_loss)

            pbar.set_postfix(**loss_data)

            if val_loss is not None and self.config.patience > 0:
                epochs_without_improvement += 1

                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    self.best_epoch = epoch
                    epochs_without_improvement = 0
                    if self.config.restore_best_weights:
                        best_state = deepcopy(self.model.state_dict())

                if epochs_without_improvement >= self.config.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch}: no val_loss "
                        f"improvement for {epochs_without_improvement} epochs"
                    )
                    pbar.set_postfix(**loss_data, status="early stopped")
                    break

            if self.config.checkpoint_dir and epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch)

        if self.config.restore_best_weights and best_state is not None:
            self.model.load_state_dict(best_state)
            logger.debug(f"Restored best model weights: val_loss={best_val_loss:.4g}")

        pbar.close()
        logger.debug("Training complete")

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

                predictions = self._forward(inputs, targets, train)
                loss = self.loss_fn(predictions, targets)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()  # type: ignore[no-untyped-call]

                    if self.config.grad_clip_norm is not None:
                        clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config.grad_clip_norm,
                        )

                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                if pbar is not None:
                    running_loss = total_loss / n_batches

                    postfix = {"train_loss": f"{running_loss:.4g}"}
                    if last_val_loss is not None:
                        postfix["val_loss"] = f"{last_val_loss:.4g}"
                        postfix["loss_gap"] = f"{abs(running_loss - last_val_loss):.4g}"  # TODO

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
