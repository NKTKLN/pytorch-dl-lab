"""Generic supervised-learning training loop for PyTorch models."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
AmpMode = Literal["auto", "off", "bf16", "fp16"]
GradNormalizer = Literal["batches", "loss_weights"]


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
        accumulation_steps: Number of consecutive batches to accumulate
            gradients over before each optimizer step, making the effective
            batch size `accumulation_steps` times the loader's. 1 steps on
            every batch.
        grad_normalizer: What the accumulated gradient is divided by before
            each optimizer step. "batches" divides by the number of
            micro-batches in the window, which matches a `loss_fn` that
            already averages over its own batch (`reduction="mean"`).
            "loss_weights" divides by the summed `loss_tracker.batch_weight`
            of the window, which matches a `loss_fn` that sums
            (`reduction="sum"`) and reproduces the gradient of one large
            batch even when batches hold unequal numbers of tokens.
        amp: Mixed-precision mode for the forward pass and loss. "auto"
            picks bf16 on CUDA devices that support it, fp16 on those that
            don't, and disables autocast on CPU. "off" runs everything in
            fp32. "bf16" and "fp16" force a dtype regardless of device;
            only fp16 needs the gradient scaler.
    """

    epochs: int = 1
    device: torch.device | str | None = None
    checkpoint_dir: str = ""
    checkpoint_every: int = 1
    restore_best_weights: bool = False
    show_progress: bool = True
    grad_clip_norm: float | None = None
    accumulation_steps: int = 1
    grad_normalizer: GradNormalizer = "batches"
    amp: AmpMode = "auto"


class Trainer:
    """Full training pipeline for a supervised PyTorch model."""

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

        Raises:
            ValueError: If `config.accumulation_steps` is less than 1, if
                `config.grad_normalizer` or `config.amp` is not a
                recognized mode, or if `config.amp` is "bf16" on a CUDA
                device without bfloat16 support.
        """
        self.config = config or TrainerConfig()

        if self.config.accumulation_steps < 1:
            raise ValueError(
                "accumulation_steps must be >= 1, got "
                f"{self.config.accumulation_steps}."
            )

        grad_normalizer: str = self.config.grad_normalizer
        if grad_normalizer not in ("batches", "loss_weights"):
            raise ValueError(
                f"Unknown grad_normalizer {grad_normalizer!r}; "
                "expected 'batches' or 'loss_weights'."
            )

        self.device = torch.device(
            self.config.device
            if self.config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.amp_enabled, self.amp_dtype = self._resolve_amp()
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.amp_enabled and self.amp_dtype is torch.float16,
        )

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.early_stopping = early_stopping
        self.loss_tracker = loss_tracker or MeanLossTracker()

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        amp_repr = (
            str(self.amp_dtype).removeprefix("torch.") if self.amp_enabled else "off"
        )
        logger.debug(
            f"Trainer initialized: model={type(model).__name__}, "
            f"optimizer={type(optimizer).__name__}, device={self.device}, "
            f"amp={amp_repr}, scaler={'on' if self.scaler.is_enabled() else 'off'}"
        )

    @staticmethod
    def _native_bf16() -> bool:
        """Return whether the current CUDA device runs bfloat16 in hardware.

        `torch.cuda.is_bf16_supported()` defaults to counting software
        emulation, which pre-Ampere cards (e.g. V100) pass while running
        bf16 far slower than the fp16 their tensor cores do support.
        """
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))

    def _resolve_amp(self) -> tuple[bool, torch.dtype]:
        """Resolve `config.amp` into an autocast (enabled, dtype) pair.

        Returns:
            Whether autocast should wrap the forward pass, and the dtype to
            run it in. The dtype is unused when autocast is disabled.

        Raises:
            ValueError: If `config.amp` is not a recognized mode, or if it
                is "bf16" on a CUDA device without native bfloat16 support.
        """
        amp: str = self.config.amp

        if amp == "auto":
            if self.device.type != "cuda":
                return False, torch.float16
            if self._native_bf16():
                return True, torch.bfloat16
            return True, torch.float16

        if amp == "off":
            return False, torch.float16

        if amp == "bf16":
            if self.device.type == "cuda" and not self._native_bf16():
                raise ValueError(
                    "amp='bf16' needs a CUDA device with native bfloat16 "
                    "support (Ampere or newer); this one would emulate it in "
                    "software. Use amp='fp16' or amp='auto'."
                )
            return True, torch.bfloat16

        if amp == "fp16":
            return True, torch.float16

        raise ValueError(
            f"Unknown amp mode {amp!r}; expected one of 'auto', 'off', 'bf16', 'fp16'."
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
            train: If True, run in training mode with gradient updates,
                stepping the optimizer once per `config.accumulation_steps`
                batches; otherwise run in evaluation mode under
                `torch.no_grad()`.
            pbar: Progress bar to update with running loss after each batch.
                If None, no progress bar is updated.

        Returns:
            The average loss over all batches.
        """
        mode = "train" if train else "eval"
        logger.debug(f"Running epoch in {mode} mode")

        self.model.train(mode=train)

        self.loss_tracker.reset()

        accumulation_steps = self.config.accumulation_steps if train else 1
        weigh_by_loss = self.config.grad_normalizer == "loss_weights"
        pending_micro_batches = 0
        window_weight = 0.0

        with torch.enable_grad() if train else torch.no_grad():
            for batch in loader:
                if train and pending_micro_batches == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                inputs, targets, *extras = (t.to(self.device) for t in batch)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    predictions = self._forward(inputs, targets, extras, train)
                    loss = self.loss_fn(predictions, targets)

                if train:
                    self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                    pending_micro_batches += 1
                    window_weight += (
                        self.loss_tracker.batch_weight(targets)
                        if weigh_by_loss
                        else 1.0
                    )

                    if pending_micro_batches == accumulation_steps:
                        self._optimizer_step(window_weight)
                        pending_micro_batches = 0
                        window_weight = 0.0

                self.loss_tracker.update(
                    loss.detach().float(), inputs, targets, extras, predictions, train
                )

                if pbar is not None:
                    running_loss = self.loss_tracker.compute()

                    postfix = {"train_loss": f"{running_loss:.4g}"}
                    if self.history["val_loss"]:
                        last_val_loss = self.history["val_loss"][-1]
                        postfix["val_loss"] = f"{last_val_loss:.4g}"

                    pbar.set_postfix(**postfix)
                    pbar.update(1)

        if pending_micro_batches > 0:
            logger.debug(
                f"Flushing {pending_micro_batches} accumulated micro-batches "
                "left over at the end of the epoch"
            )
            self._optimizer_step(window_weight)

        avg_loss = self.loss_tracker.compute()
        logger.debug(f"Epoch {mode} pass: avg_loss={avg_loss:.4f}")
        return avg_loss

    def _optimizer_step(self, window_weight: float) -> None:
        """Normalize and clip the accumulated gradients, then step the optimizer.

        Args:
            window_weight: Summed weight of the micro-batches accumulated
                since the last step, as chosen by `config.grad_normalizer`.
                Dividing by it commutes with the scaler's own factor, so it
                is safe to apply while the gradients are still scaled.
        """
        if window_weight <= 0.0:
            logger.warning(
                "Skipping optimizer step: accumulation window weighs "
                f"{window_weight}, which cannot normalize the gradients"
            )
            return

        if window_weight != 1.0:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(window_weight)

        if self.config.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip_norm,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

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
                "scaler_state_dict": self.scaler.state_dict(),
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

        scaler_state = state.get("scaler_state_dict")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        history = state.get("history")
        if history is not None:
            self.history = history

        epoch = int(state.get("epoch", 0))
        logger.debug(f"Loaded trainer state: {path}, epoch={epoch}")
        return epoch
