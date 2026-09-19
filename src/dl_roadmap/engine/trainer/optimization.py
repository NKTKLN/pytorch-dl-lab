"""Owns the optimizer step: AMP, accumulation, clipping and scheduling."""

from dataclasses import dataclass
from typing import Any, Literal, get_args

import torch
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dl_roadmap.engine.trainer.schedulers import WarmupScheduler, step_scheduler

AmpMode = Literal["auto", "off", "bf16", "fp16"]
GradNormalizer = Literal["batches", "loss_weights"]


@dataclass
class OptimizationConfig:
    """Configure the optimizer step.

    Attributes:
        grad_clip_norm: Maximum gradient norm before each optimizer step;
            must be > 0. None disables gradient clipping.
        accumulation_steps: Number of batches per gradient accumulation
            window. Any remaining batches are processed at epoch end.
        grad_normalizer: Normalize accumulated gradients by batch count
            ("batches", for mean-reduced loss) or summed weights from
            `loss_tracker.batch_weight` ("loss_weights", for sum-reduced loss).
        amp: Autocast mode. "auto" selects native BF16 when supported on CUDA,
            otherwise FP16, and disables autocast on non-CUDA devices.
            "off" disables autocast. "bf16" and "fp16" select an explicit dtype;
            BF16 on CUDA requires native hardware support.
    """

    grad_clip_norm: float | None = None
    accumulation_steps: int = 1
    grad_normalizer: GradNormalizer = "batches"
    amp: AmpMode = "auto"

    def __post_init__(self) -> None:
        """Validate the optimizer-step options.

        Raises:
            ValueError: If `grad_clip_norm` is nonpositive, `accumulation_steps`
                is below 1, or `grad_normalizer` or `amp` is unrecognized.
        """
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0 or None.")

        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1.")

        if self.grad_normalizer not in get_args(GradNormalizer):
            raise ValueError(f"Unknown grad_normalizer {self.grad_normalizer!r}.")

        if self.amp not in get_args(AmpMode):
            raise ValueError(f"Unknown amp mode {self.amp!r}.")


class OptimizationEngine:
    """Run backward passes and optimizer steps for a trainer."""

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler | WarmupScheduler | None = None,
        config: OptimizationConfig | None = None,
    ) -> None:
        """Bind the optimizer and scheduler to their step settings.

        Args:
            optimizer: Optimizer bound to the model parameters.
            scheduler: Optional learning-rate scheduler. Standard schedulers
                step after each epoch; `WarmupScheduler` also receives updates
                after successful optimizer steps. `ReduceLROnPlateau` requires
                validation loss.
            config: Optimizer-step options. None creates a default
                `OptimizationConfig`.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or OptimizationConfig()

        self.device: torch.device | None = None
        self.amp_enabled = False
        self.amp_dtype = torch.float16
        self._scaler: torch.amp.GradScaler | None = None

        self._pending_micro_batches = 0
        self._window_weight = 0.0

    @property
    def scaler(self) -> torch.amp.GradScaler:
        """Return the gradient scaler, which exists once the device is known.

        Returns:
            torch.amp.GradScaler: Scaler created by `prepare()`.

        Raises:
            RuntimeError: If `prepare()` has not been called yet.
        """
        if self._scaler is None:
            raise RuntimeError(
                "OptimizationEngine is not prepared: call prepare(device) first. "
                "BaseTrainer does this for you."
            )

        return self._scaler

    @property
    def parameters(self) -> list[torch.Tensor]:
        """Return every parameter the optimizer updates."""
        return [p for group in self.optimizer.param_groups for p in group["params"]]

    def prepare(self, device: torch.device) -> "OptimizationEngine":
        """Resolve autocast for `device` and create the gradient scaler.

        Args:
            device: Device the training runs on; selects the AMP dtype.

        Returns:
            OptimizationEngine: This engine, for chaining.
        """
        if self._scaler is not None and self.device == device:
            return self

        self.device = device
        self.amp_enabled, self.amp_dtype = self._resolve_amp(device)
        self._scaler = torch.amp.GradScaler(
            device.type,
            enabled=self.amp_enabled and self.amp_dtype is torch.float16,
        )

        logger.debug(
            f"Optimization prepared: device={device}, "
            f"amp={str(self.amp_dtype) if self.amp_enabled else 'off'}, "
            f"scaler={'on' if self._scaler.is_enabled() else 'off'}"
        )

        return self

    def _resolve_amp(self, device: torch.device) -> tuple[bool, torch.dtype]:
        """Resolve the autocast state and dtype from `config.amp`.

        Args:
            device: Device the training runs on.

        Returns:
            tuple[bool, torch.dtype]: Whether autocast is enabled and its dtype.
                The dtype is unused when autocast is disabled.

        Raises:
            ValueError: If explicit BF16 is requested on CUDA without native
                bfloat16 support.
        """
        amp: str = self.config.amp

        if amp == "auto":
            if device.type != "cuda":
                return False, torch.float16
            if torch.cuda.is_bf16_supported(including_emulation=False):
                return True, torch.bfloat16
            return True, torch.float16

        if amp == "off":
            return False, torch.float16

        if amp == "bf16":
            if device.type == "cuda" and not torch.cuda.is_bf16_supported(
                including_emulation=False
            ):
                raise ValueError(
                    "amp='bf16' requires native bfloat16 support on CUDA. "
                    "Use amp='fp16' or amp='auto'."
                )
            return True, torch.bfloat16

        return True, torch.float16

    def autocast(self) -> torch.amp.autocast:
        """Return the autocast context for the forward pass.

        Returns:
            torch.amp.autocast: Context manager, disabled when AMP is off.
        """
        device_type = self.device.type if self.device is not None else "cpu"

        return torch.amp.autocast(
            device_type=device_type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    def backward(self, loss: torch.Tensor, batch_weight: float = 1.0) -> None:
        """Scale the loss, back-propagate it and open the accumulation window.

        Args:
            loss: Scalar loss for the micro-batch just processed.
            batch_weight: Weight this micro-batch contributes to the window,
                as chosen by `config.grad_normalizer`.
        """
        if self._pending_micro_batches == 0:
            self.optimizer.zero_grad(set_to_none=True)

        self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        self._pending_micro_batches += 1
        self._window_weight += batch_weight

    def step_if_ready(self) -> bool:
        """Step the optimizer once the accumulation window is full.

        Returns:
            bool: True if the optimizer stepped.
        """
        if self._pending_micro_batches < self.config.accumulation_steps:
            return False

        self._step()
        return True

    def flush(self) -> bool:
        """Step on micro-batches left over at the end of an epoch.

        Returns:
            bool: True if the optimizer stepped.
        """
        if self._pending_micro_batches == 0:
            return False

        logger.debug(
            f"Flushing {self._pending_micro_batches} accumulated micro-batches "
            "left over at the end of the epoch"
        )
        self._step()
        return True

    def _step(self) -> None:
        """Normalize and clip gradients, then step the optimizer and warmup."""
        window_weight = self._window_weight
        self._pending_micro_batches = 0
        self._window_weight = 0.0

        if window_weight <= 0.0:
            logger.warning(
                "Skipping optimizer step: accumulation window weighs "
                f"{window_weight}, which cannot normalize the gradients"
            )
            return

        if window_weight != 1.0:
            for param in self.parameters:
                if param.grad is not None:
                    param.grad.div_(window_weight)

        if self.config.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.parameters, max_norm=self.config.grad_clip_norm)

        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        stepped = self.scaler.get_scale() >= scale_before

        if stepped and isinstance(self.scheduler, WarmupScheduler):
            self.scheduler.step_batch()

    def step_epoch(self, val_loss: float | None = None) -> None:
        """Advance the learning-rate scheduler at epoch end.

        Args:
            val_loss: Validation loss for `ReduceLROnPlateau`. Other schedulers
                do not require it.

        Raises:
            ValueError: If the scheduler requires validation loss but
                `val_loss` is None.
        """
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, WarmupScheduler):
            self.scheduler.step_epoch(val_loss)
            return

        step_scheduler(self.scheduler, val_loss)

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer, scheduler and scaler state.

        Returns:
            dict[str, Any]: State under "optimizer", "scheduler" and "scaler";
                the scheduler entry is None when no scheduler is configured.
        """
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler": self.scaler.state_dict() if self._scaler is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimizer, scheduler and scaler state.

        Args:
            state: Mapping written by `state_dict()`. Missing entries are
                skipped, so states saved without a scheduler still load.
        """
        self.optimizer.load_state_dict(state["optimizer"])

        scheduler_state = state.get("scheduler")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = state.get("scaler")
        if scaler_state is not None and self._scaler is not None:
            self._scaler.load_state_dict(scaler_state)
