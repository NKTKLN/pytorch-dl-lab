"""Orchestrate training: run the loop and delegate the work to collaborators."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
from loguru import logger
from torch import nn
from tqdm import tqdm

from dl_roadmap.engine.trainer.config import TrainingConfig
from dl_roadmap.engine.trainer.context import Phase, StepContext
from dl_roadmap.engine.trainer.early_stopping import EarlyStopping
from dl_roadmap.engine.trainer.metrics_manager import MetricsManager
from dl_roadmap.engine.trainer.optimization import NoOptimization, Optimization
from dl_roadmap.engine.trainer.state_store import TrainerStateStore

Batch = tuple[torch.Tensor, ...]
"""One batch: inputs, then targets, then any extra tensors the model needs.

The trainer reads the first two positions and hands the batch to `_forward`
whole, so a collate function is free to append its own tensors — a decoder
input, a mask — and a `NamedTuple` batch keeps its field names all the way
into `_forward`.
"""

PairBatch = tuple[torch.Tensor, torch.Tensor]
"""The common case: inputs and targets, nothing else."""
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
EpochCallback = Callable[[int, float, float | None], None]

BAR_FORMAT = (
    "{desc}{percentage:3.0f}%[{bar:20}] {n_fmt}/{total_fmt} :: eta={remaining}{postfix}"
)


class BaseTrainer[BatchT: Batch]:
    """Train and evaluate supervised PyTorch models."""

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        loss_fn: LossFn,
        config: TrainingConfig | None = None,
        optimization: Optimization | None = None,
        metrics: MetricsManager | None = None,
        early_stopping: EarlyStopping | None = None,
        callbacks: list[EpochCallback] | None = None,
    ) -> None:
        """Wire the model to the components that train and measure it.

        Args:
            model: Model to train; moved to `config.device`.
            loss_fn: Compute a scalar loss from predictions and targets.
            config: Training lifecycle options. None creates a default
                `TrainingConfig`.
            optimization: Engine owning the optimizer step. None installs
                `NoOptimization`, leaving the trainer in evaluation mode:
                `evaluate` and `predict` work, `fit` raises.
            metrics: Loss tracker and metrics. None creates a default
                `MetricsManager` averaging batch losses.
            early_stopping: Strategy selecting the best epoch and deciding
                when to stop. Requires a validation loader.
            callbacks: Functions called after each epoch's scheduler update
                with `(epoch, train_loss, val_loss)`. Epochs start at 1;
                `val_loss` is None when validation is disabled.
        """
        self.config = config or TrainingConfig()
        self.device = self.config.resolve_device()

        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimization = optimization or NoOptimization()
        self.metrics = metrics or MetricsManager()
        self.early_stopping = early_stopping
        self.callbacks = callbacks or []

        self.optimization.prepare(self.device)

        self.state_store = TrainerStateStore(
            self.model, self.optimization, self.config.checkpoint_dir
        )

        logger.debug(
            f"Trainer initialized: model={type(model).__name__}, "
            f"device={self.device}, "
            f"optimization={'on' if self.optimization.can_step else 'off'}"
        )

    @property
    def history(self) -> dict[str, list[float]]:
        """Return the training history recorded so far."""
        return self.state_store.history

    def _forward(self, batch: BatchT, _ctx: StepContext) -> torch.Tensor:
        """Compute model predictions for a batch.

        Args:
            batch: Every tensor of the batch, already on `self.device`, with
                its own type — field names included for a `NamedTuple` batch.
            _ctx: Phase, epoch and step the batch belongs to; unused here.

        Returns:
            torch.Tensor: Model predictions passed to `self.loss_fn`.
        """
        return self.model(batch[0])  # type: ignore[no-any-return]

    def _targets(self, batch: BatchT) -> torch.Tensor | None:
        """Return the tensor `loss_fn` and the metrics compare against.

        Args:
            batch: Every tensor of the batch, already on `self.device`.

        Returns:
            torch.Tensor | None: The second tensor, or None for a batch of
                inputs alone, as `predict` allows.
        """
        return batch[1] if len(batch) > 1 else None

    def _to_device(self, batch: BatchT) -> BatchT:
        """Move every tensor of a batch to `self.device`, keeping its type.

        Args:
            batch: Batch as the loader yielded it.

        Returns:
            BatchT: The same kind of batch, tensor by tensor on the device.
                A `NamedTuple` is rebuilt through `_make`, so it keeps its
                field names; anything else comes back a plain tuple.
        """
        moved = [tensor.to(self.device) for tensor in batch]
        make = getattr(batch, "_make", None)

        if make is not None:
            return cast(BatchT, make(moved))

        return cast(BatchT, tuple(moved))

    def fit(
        self,
        train_loader: Iterable[BatchT],
        val_loader: Iterable[BatchT] | None = None,
    ) -> None:
        """Run the training loop for `config.epochs` epochs.

        Args:
            train_loader: Batches of (inputs, targets, *extras) used for training.
            val_loader: Optional batches of (inputs, targets, *extras) used
                for per-epoch validation. Required if `early_stopping` is set.

        Raises:
            ValueError: If no `optimization` engine is configured, if
                `early_stopping` is set but no `val_loader` is given, or if
                `scheduler` is a `ReduceLROnPlateau` and no `val_loader` is given.
        """
        if not self.optimization.can_step:
            raise ValueError("fit requires an OptimizationEngine.")

        if self.early_stopping is not None and val_loader is None:
            raise ValueError("early_stopping requires a val_loader.")

        logger.debug(
            f"Starting training: epochs={self.config.epochs}, "
            f"val={'yes' if val_loader is not None else 'no'}, "
            f"checkpoint_dir='{self.config.checkpoint_dir or 'disabled'}'"
        )

        batches_per_epoch = len(train_loader)  # type: ignore[arg-type]
        total_steps = batches_per_epoch and self.config.epochs * batches_per_epoch
        epoch_width = len(str(self.config.epochs))

        pbar = tqdm(
            total=total_steps,
            desc=f"epoch {1:>{epoch_width}}/{self.config.epochs}",
            ascii=" >=",
            bar_format=BAR_FORMAT,
            leave=True,
            disable=not self.config.show_progress,
        )

        best_state: dict[str, Any] | None = None

        try:
            for epoch in range(1, self.config.epochs + 1):
                pbar.set_description(
                    f"epoch {epoch:>{epoch_width}}/{self.config.epochs}"
                )

                train_loss, train_metrics = self._run_epoch(
                    train_loader, phase="train", epoch=epoch, pbar=pbar
                )
                self.history["train_loss"].append(train_loss)

                loss_data = {"train_loss": f"{train_loss:.4g}"}

                val_loss: float | None = None
                val_metrics: dict[str, float] | None = None
                if val_loader is not None:
                    val_loss, val_metrics = self._run_epoch(
                        val_loader, phase="val", epoch=epoch
                    )
                    self.history["val_loss"].append(val_loss)
                    loss_data["val_loss"] = f"{val_loss:.4g}"

                loss_data.update(
                    self.metrics.record(self.history, train_metrics, val_metrics)
                )

                self.optimization.step_epoch(val_loss)

                for callback in self.callbacks:
                    callback(epoch, train_loss, val_loss)

                pbar.set_postfix(loss_data)

                if self.early_stopping is not None:
                    self.early_stopping.update(
                        epoch, train_loss, val_loss, self.history
                    )

                    if self.early_stopping.is_best and self.config.restore_best_weights:
                        best_state = deepcopy(self.model.state_dict())

                    if self.early_stopping.should_stop:
                        pbar.set_postfix(loss_data | {"status": "early stopped"})
                        break

                if (
                    self.config.checkpoint_dir
                    and epoch % self.config.checkpoint_every == 0
                ):
                    self.save_checkpoint(epoch)

            if self.config.restore_best_weights and best_state is not None:
                self.model.load_state_dict(best_state)
                logger.debug(
                    "Restored best model weights from epoch "
                    f"{self.early_stopping.best_epoch}"  # type: ignore[union-attr]
                )
        finally:
            pbar.close()

        logger.debug("Training complete")

    def evaluate(self, loader: Iterable[BatchT]) -> tuple[float, dict[str, float]]:
        """Run one evaluation pass, without touching the optimizer.

        Args:
            loader: Batches of (inputs, targets, *extras) to evaluate on.

        Returns:
            tuple[float, dict[str, float]]: The average loss over the pass and
                each configured metric's value.
        """
        return self._run_epoch(loader, phase="val")

    def predict(self, loader: Iterable[BatchT]) -> torch.Tensor:
        """Collect model predictions for every batch in `loader`.

        Args:
            loader: Batches whose first tensor holds the inputs. Any further
                tensors are passed on as extras; no targets are read, so a
                loader of inputs alone works.

        Returns:
            torch.Tensor: Predictions for all batches, concatenated on CPU.
        """
        self.model.eval()
        ctx = StepContext(phase="predict", step=self.optimization.steps)
        outputs: list[torch.Tensor] = []

        with torch.no_grad():
            for raw_batch in loader:
                batch = self._to_device(raw_batch)

                with self.optimization.autocast():
                    predictions = self._forward(batch, ctx)

                outputs.append(predictions.detach().float().cpu())

        return torch.cat(outputs)

    def save_checkpoint(self, epoch: int) -> Path:
        """Save the trainer state into `config.checkpoint_dir`.

        Args:
            epoch: Current epoch number, used to name the checkpoint file.

        Returns:
            Path: The path the checkpoint was written to.
        """
        return self.state_store.save_checkpoint(epoch)

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Restore the trainer state from a checkpoint file.

        Args:
            checkpoint_path: Path to a file written by `save_checkpoint`.

        Returns:
            int: The epoch number the checkpoint was saved at.
        """
        return self.state_store.load(checkpoint_path, map_location=self.device)

    def save(self, path: str | Path, epoch: int = 0) -> Path:
        """Save the full trainer state to a single file.

        Args:
            path: File path to write the state to.
            epoch: Epoch to record alongside the state.

        Returns:
            Path: The path the state was written to.
        """
        return self.state_store.save(path, epoch)

    def load(self, path: str | Path) -> int:
        """Restore the full trainer state from a file written by `save`.

        Args:
            path: File path to load the state from.

        Returns:
            int: The epoch number recorded in the saved state.
        """
        return self.state_store.load(path, map_location=self.device)

    def _run_epoch(
        self,
        loader: Iterable[BatchT],
        phase: Phase,
        epoch: int = 0,
        pbar: Any | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Run a single train or evaluation pass over `loader`.

        Args:
            loader: Batches of (inputs, targets, *extras).
            phase: "train" runs with gradient updates, "val" runs under
                `torch.no_grad()`.
            epoch: Epoch this pass belongs to, counted from 1; 0 outside `fit`.
            pbar: Progress bar to update with running loss after each batch.
                If None, no progress bar is updated.

        Returns:
            tuple[float, dict[str, float]]: The average loss over all batches,
                and each configured metric's value over the same pass.

        Raises:
            ValueError: If a batch carries no targets to compare against.
        """
        is_training = phase == "train"
        logger.debug(f"Running epoch in {phase} mode")

        self.model.train(mode=is_training)
        self.metrics.reset()

        optimization = self.optimization
        weigh_by_loss = (
            is_training and optimization.config.grad_normalizer == "loss_weights"
        )

        with torch.enable_grad() if is_training else torch.no_grad():
            for raw_batch in loader:
                batch = self._to_device(raw_batch)
                targets = self._targets(batch)

                if targets is None:
                    raise ValueError(
                        f"{phase} batches need targets; got a batch of "
                        f"{len(batch)} tensor(s)."
                    )

                ctx = StepContext(
                    phase=phase,
                    epoch=epoch,
                    epochs=self.config.epochs,
                    step=self.optimization.steps,
                )

                with self.optimization.autocast():
                    predictions = self._forward(batch, ctx)
                    loss = self.loss_fn(predictions, targets)

                if is_training:
                    weight = (
                        self.metrics.batch_weight(targets) if weigh_by_loss else 1.0
                    )
                    optimization.backward(loss, weight)
                    optimization.step_if_ready()

                self.metrics.update(
                    loss.detach().float(),
                    batch[0],
                    targets,
                    list(batch[2:]),
                    predictions.detach(),
                    ctx,
                )

                if pbar is not None:
                    pbar.set_postfix(self.metrics.running_postfix(self.history))
                    pbar.update(1)

        if is_training:
            optimization.flush()

        avg_loss = self.metrics.loss()
        values = self.metrics.values()

        metrics_repr = "".join(
            f", {name}={value:.4f}" for name, value in values.items()
        )
        logger.debug(f"Epoch {phase} pass: avg_loss={avg_loss:.4f}{metrics_repr}")

        return avg_loss, values
