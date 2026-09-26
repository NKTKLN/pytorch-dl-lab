"""Own the optimizer step: AMP, accumulation, clipping and scheduling."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal, get_args

import torch
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dl_roadmap.engine.trainer.lr_schedulers import LRSchedule, as_lr_schedule

AmpMode = Literal["auto", "off", "bf16", "fp16"]
"""How the forward pass is autocast; see `OptimizationConfig.amp`."""

GradNormalizer = Literal["batches", "loss_weights"]
"""What accumulated gradients are divided by; see `OptimizationConfig`."""


@dataclass
class OptimizationConfig:
    """Configure the optimizer step.

    Attributes:
        grad_clip_norm: Maximum gradient norm before each optimizer step;
            must be > 0. None disables gradient clipping.
        accumulation_steps: Number of batches per gradient accumulation
            window. A window a training pass leaves incomplete is stepped at
            the end of that pass, so no gradients carry over.
        grad_normalizer: Normalize accumulated gradients by batch count
            ("batches", for mean-reduced loss) or summed weights from
            `loss_tracker.batch_weight` ("loss_weights", for sum-reduced loss);
            must match the trainer's `loss_tracker`.
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


class Optimization(ABC):
    """What a trainer needs from whatever drives the optimizer step.

    Attributes:
        can_step: Whether this implementation updates the model at all.
            `fit` refuses to run without it.
        config: Settings the trainer reads, such as `grad_normalizer`.
        steps: Optimizer steps that updated the weights so far; a step the
            gradient scaler skips is not counted.
    """

    can_step: bool
    config: OptimizationConfig
    steps: int

    @abstractmethod
    def prepare(self, device: torch.device) -> "Optimization":
        """Bind the implementation to the device training runs on.

        Args:
            device: Device the model and batches live on.

        Returns:
            Optimization: This object, for chaining.
        """
        raise NotImplementedError

    @abstractmethod
    def autocast(self) -> AbstractContextManager[None]:
        """Return the context the forward pass runs in.

        Returns:
            AbstractContextManager[None]: Autocast context, or a no-op one.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, loss: torch.Tensor, batch_weight: float = 1.0) -> None:
        """Back-propagate one micro-batch into the accumulation window.

        Args:
            loss: Scalar loss for the micro-batch just processed.
            batch_weight: Weight this micro-batch adds to the window; the
                accumulated gradients are divided by the summed weights
                when the window is stepped.
        """
        raise NotImplementedError

    @abstractmethod
    def step_if_ready(self) -> bool:
        """Step the optimizer once the accumulation window is full.

        Returns:
            bool: True if the window was full and has been closed. The
                weights may still be left unchanged, as `steps` shows.
        """
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> bool:
        """Step on micro-batches left over at the end of a training pass.

        Returns:
            bool: True if a window was open and has been closed. The weights
                may still be left unchanged, as `steps` shows.
        """
        raise NotImplementedError

    @abstractmethod
    def step_interval(self, values: Mapping[str, float]) -> None:
        """Advance the learning-rate schedule once an interval is recorded.

        Args:
            values: What the interval recorded, for schedules that watch it.
        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return the state a checkpoint has to carry.

        Returns:
            dict[str, Any]: State to hand back to `load_state_dict`.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state written by `state_dict`.

        Args:
            state: Mapping returned by `state_dict`.
        """
        raise NotImplementedError


class NoOptimization(Optimization):
    """Stand-in for a trainer that only evaluates or predicts."""

    can_step = False

    def __init__(self) -> None:
        """Start with default settings and no steps to take."""
        self.config = OptimizationConfig()
        self.steps = 0

    def prepare(self, _device: torch.device) -> "NoOptimization":
        """Ignore the device, since nothing here depends on it.

        Args:
            _device: Device the model and batches live on; unused.

        Returns:
            NoOptimization: This object, for chaining.
        """
        return self

    def autocast(self) -> AbstractContextManager[None]:
        """Return a no-op context, leaving the forward pass in full precision.

        Returns:
            AbstractContextManager[None]: A context that does nothing.
        """
        return nullcontext()

    def backward(self, _loss: torch.Tensor, _batch_weight: float = 1.0) -> None:
        """Refuse to back-propagate, since there is no optimizer.

        Args:
            _loss: Scalar loss for the micro-batch just processed; unused.
            _batch_weight: Weight this micro-batch would contribute; unused.

        Raises:
            RuntimeError: Always; reaching this means a training pass ran
                without an `OptimizationEngine`.
        """
        raise RuntimeError("Cannot train without an OptimizationEngine.")

    def step_if_ready(self) -> bool:
        """Report that no optimizer stepped.

        Returns:
            bool: Always False.
        """
        return False

    def flush(self) -> bool:
        """Report that no optimizer stepped.

        Returns:
            bool: Always False.
        """
        return False

    def step_interval(self, _values: Mapping[str, float]) -> None:
        """Do nothing, since there is no schedule to advance.

        Args:
            _values: What the interval recorded; unused.
        """

    def state_dict(self) -> dict[str, Any]:
        """Return nothing to save.

        Returns:
            dict[str, Any]: An empty mapping.
        """
        return {}

    def load_state_dict(self, _state: dict[str, Any]) -> None:
        """Ignore saved optimizer state.

        Args:
            _state: Mapping returned by `state_dict`; unused.
        """


class OptimizationEngine(Optimization):
    """Run backward passes and optimizer steps for a trainer."""

    can_step = True

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: LRSchedule | LRScheduler | None = None,
        config: OptimizationConfig | None = None,
    ) -> None:
        """Bind the optimizer and scheduler to their step settings.

        Args:
            optimizer: Optimizer bound to the model parameters.
            scheduler: Optional learning-rate schedule. A bare PyTorch
                scheduler advances after every interval; wrap it in
                `TorchLRSchedule` to advance per optimizer step or to choose
                what a `ReduceLROnPlateau` monitors, or in `Warmup` to ramp
                up first.
            config: Optimizer-step options. None creates a default
                `OptimizationConfig`.
        """
        self.optimizer = optimizer
        self.scheduler = as_lr_schedule(scheduler) if scheduler is not None else None
        self.config = config or OptimizationConfig()

        self.device: torch.device | None = None
        self.amp_enabled = False
        self.amp_dtype = torch.float16
        self._scaler: torch.amp.GradScaler | None = None

        self._pending_micro_batches = 0
        self._window_weight = 0.0

        self.steps = 0

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
                "Trainer does this for you."
            )

        return self._scaler

    @property
    def parameters(self) -> list[torch.Tensor]:
        """Return every parameter the optimizer updates.

        Returns:
            list[torch.Tensor]: Parameters of all param groups, in order.
        """
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
        """Scale the loss and back-propagate it into the accumulation window.

        The first micro-batch of a window clears the previous gradients.

        Args:
            loss: Scalar loss for the micro-batch just processed.
            batch_weight: Weight this micro-batch adds to the window: 1.0,
                or the loss tracker's weight when `config.grad_normalizer`
                is "loss_weights".
        """
        if self._pending_micro_batches == 0:
            self.optimizer.zero_grad(set_to_none=True)

        self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        self._pending_micro_batches += 1
        self._window_weight += batch_weight

    def step_if_ready(self) -> bool:
        """Step the optimizer once the accumulation window is full.

        Returns:
            bool: True if the window was full and has been closed. The
                weights may still be left unchanged, as `steps` shows.
        """
        if self._pending_micro_batches < self.config.accumulation_steps:
            return False

        self._step()
        return True

    def flush(self) -> bool:
        """Step on micro-batches left over at the end of a training pass.

        Returns:
            bool: True if a window was open and has been closed. The weights
                may still be left unchanged, as `steps` shows.
        """
        if self._pending_micro_batches == 0:
            return False

        logger.debug(
            f"Flushing {self._pending_micro_batches} accumulated micro-batches "
            "left over at the end of the pass"
        )
        self._step()
        return True

    def _step(self) -> None:
        """Close the window: normalize, clip and step, then advance the schedule.

        The step is skipped when the window weighs nothing, and the gradient
        scaler skips it on inf or NaN gradients; either way neither `steps`
        nor the learning-rate schedule advances.
        """
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

        if not stepped:
            return

        self.steps += 1

        if self.scheduler is not None:
            self.scheduler.after_step()

    def step_interval(self, values: Mapping[str, float]) -> None:
        """Advance the learning-rate schedule once an interval is recorded.

        Args:
            values: What the interval recorded; a `ReduceLROnPlateau` steps
                on the value it monitors.

        Raises:
            ValueError: If the schedule monitors a value the interval did
                not record.
        """
        if self.scheduler is not None:
            self.scheduler.after_interval(values)

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer, schedule and scaler state, and the step count.

        Returns:
            dict[str, Any]: State under "optimizer", "scheduler", "scaler"
                and "steps"; the scheduler and scaler entries are None when
                there is no schedule or the engine is not prepared yet.
        """
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler": self.scaler.state_dict() if self._scaler is not None else None,
            "steps": self.steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimizer, schedule and scaler state, and the step count.

        The learning rate comes back with the optimizer's state.

        Args:
            state: Mapping written by `state_dict()`. Only "optimizer" is
                required; missing entries are skipped, so states saved
                without a schedule still load. The scaler state is applied
                only once the engine is prepared.
        """
        self.optimizer.load_state_dict(state["optimizer"])

        scheduler_state = state.get("scheduler")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = state.get("scaler")
        if scaler_state is not None and self._scaler is not None:
            self._scaler.load_state_dict(scaler_state)

        self.steps = int(state.get("steps", 0))
