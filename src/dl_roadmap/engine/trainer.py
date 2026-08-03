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

from dl_roadmap.engine.early_stopping import EarlyStopping
from dl_roadmap.engine.loss_tracker import LossTracker, MeanLossTracker

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
    """

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: LossFn,
        scheduler: LRScheduler | None = None,
        config: TrainerConfig | None = None,
        callbacks: list[EpochCallback] | None = None,
        early_stopping: EarlyStopping | None = None,
        loss_tracker: LossTracker | None = None,
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
            early_stopping: Optional strategy deciding when to stop training
                and which epoch counts as best. Requires `fit` to be called
                with a `val_loader`.
            loss_tracker: Strategy for aggregating per-batch loss into the
                epoch's reported loss. Defaults to `MeanLossTracker`, i.e. a
                plain per-batch average.
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
        self.early_stopping = early_stopping
        self.loss_tracker = loss_tracker or MeanLossTracker()

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
        self,
        inputs: torch.Tensor,
        _targets: torch.Tensor,
        _extras: list[torch.Tensor],
        _train: bool,
    ) -> torch.Tensor:
        """Computes model predictions for a batch.

        Override to pass extra context into the model's forward call, e.g.
        ground-truth targets for teacher forcing in a seq2seq decoder.
        Ignored here; the base implementation only needs `inputs`.

        Args:
            inputs: Batch inputs, already moved to `self.device`.
            _targets: Batch targets, already moved to `self.device`.
            _extras: Any batch elements beyond inputs/targets, already moved
                to `self.device`.
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
                per-epoch validation. Required if `early_stopping` is set.

        Raises:
            ValueError: If `early_stopping` is set but no `val_loader` is
                given, or if `scheduler` is a `ReduceLROnPlateau` and no
                `val_loader` is given.
        """
        if self.early_stopping is not None and val_loader is None:
            raise ValueError("early_stopping requires a val_loader.")

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

        best_state: dict[str, Any] | None = None

        for epoch in range(1, self.config.epochs + 1):
            pbar.set_description(f"epoch {epoch:>{epoch_width}}/{self.config.epochs}")

            train_loss = self._run_epoch(
                train_loader,
                train=True,
                pbar=pbar,
            )
            self.history["train_loss"].append(train_loss)

            loss_data = {"train_loss": f"{train_loss:.4g}"}

            val_loss: float | None = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, train=False)
                self.history["val_loss"].append(val_loss)
                loss_data["val_loss"] = f"{val_loss:.4g}"

            self._step_scheduler(val_loss)

            for callback in self.callbacks:
                callback(epoch, train_loss, val_loss)

            pbar.set_postfix(**loss_data)

            if self.early_stopping is not None:
                self.early_stopping.update(epoch, train_loss, val_loss, self.history)

                if self.early_stopping.is_best and self.config.restore_best_weights:
                    best_state = deepcopy(self.model.state_dict())

                if self.early_stopping.should_stop:
                    pbar.set_postfix(**loss_data, status="early stopped")
                    break

            if self.config.checkpoint_dir and epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch)

        if self.config.restore_best_weights and best_state is not None:
            self.model.load_state_dict(best_state)
            logger.debug(
                "Restored best model weights from epoch "
                f"{self.early_stopping.best_epoch}"
            )

        pbar.close()
        logger.debug("Training complete")

    def _run_epoch(
        self,
        loader: Iterable[Batch],
        train: bool,
        pbar: Any | None = None,
    ) -> float:
        """Run a single train or evaluation pass over `loader`.

        Args:
            loader: Batches of (inputs, targets).
            train: If True, run in training mode with gradient updates;
                otherwise run in evaluation mode under `torch.no_grad()`.
            pbar: Progress bar to update with running loss after each batch.
                If None, no progress bar is updated.

        Returns:
            The average loss over all batches.
        """
        mode = "train" if train else "eval"
        logger.debug(f"Running epoch in {mode} mode")
        self.model.train(mode=train)

        self.loss_tracker.reset()

        with torch.enable_grad() if train else torch.no_grad():
            for batch in loader:
                inputs, targets, *extras = (t.to(self.device) for t in batch)

                predictions = self._forward(inputs, targets, extras, train)
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

                self.loss_tracker.update(
                    loss, inputs, targets, extras, predictions, train
                )

                if pbar is not None:
                    running_loss = self.loss_tracker.compute()

                    postfix = {"train_loss": f"{running_loss:.4g}"}
                    if self.history["val_loss"]:
                        last_val_loss = self.history["val_loss"][-1]
                        postfix["val_loss"] = f"{last_val_loss:.4g}"

                    pbar.set_postfix(**postfix)
                    pbar.update(1)

        avg_loss = self.loss_tracker.compute()
        logger.debug(f"Epoch {mode} pass: avg_loss={avg_loss:.4f}")
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

        self.save(checkpoint_path, epoch)
        logger.debug(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Restore model and optimizer state from a checkpoint file.

        Args:
            checkpoint_path: Path to a checkpoint file written by `save_checkpoint`.

        Returns:
            The epoch number the checkpoint was saved at.
        """
        epoch = self.load(checkpoint_path)
        logger.debug(f"Loaded checkpoint: {checkpoint_path}, epoch={epoch}")
        return epoch

    def save(self, path: str | Path, epoch: int = 0) -> Path:
        """Save the full trainer state to a single file.

        Unlike `save_checkpoint`, this also persists the scheduler and
        training history, so training can be resumed exactly where it
        left off (e.g. across a notebook restart), not just the weights.

        Args:
            path: File path to write the state to. Parent directories are
                created if they don't exist.
            epoch: Epoch to record alongside the state. Defaults to 0 for
                ad-hoc saves outside of `fit`.

        Returns:
            The path the state was written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "history": self.history,
            },
            path,
        )
        logger.debug(f"Saved trainer state: {path}")

        return path

    def load(self, path: str | Path) -> int:
        """Restore the full trainer state from a file written by `save`.

        Args:
            path: File path to load the state from.

        Returns:
            The epoch number recorded in the saved state.

        Raises:
            FileNotFoundError: If no file exists at `path`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Trainer state not found: {path}")

        state: dict[str, Any] = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        scheduler_state = state.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        history = state.get("history")
        if history is not None:
            self.history = history

        epoch = int(state.get("epoch", 0))
        logger.debug(f"Loaded trainer state: {path}, epoch={epoch}")
        return epoch
