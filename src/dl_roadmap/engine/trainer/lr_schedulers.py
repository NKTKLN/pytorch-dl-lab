"""When learning-rate schedulers advance, and the warmup ahead of them."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch.optim.lr_scheduler import LinearLR, LRScheduler, ReduceLROnPlateau

Cadence = Literal["step", "interval"]
"""When a schedule advances: after each optimizer step that updated the
weights, or after each interval of the run once its results are recorded."""


class LRSchedule(ABC):
    """Advance learning rates as a run takes steps and finishes intervals."""

    @abstractmethod
    def after_step(self) -> None:
        """React to an optimizer step that updated the weights."""
        raise NotImplementedError

    @abstractmethod
    def after_interval(self, values: Mapping[str, float]) -> None:
        """React to a finished interval.

        Args:
            values: What the interval recorded, under the keys the history
                stores it by: "train_loss", "val_loss", "val_<metric>"...
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
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the state `state_dict` returned.

        Args:
            state_dict: A mapping as returned by `state_dict`.
        """
        raise NotImplementedError


class TorchLRSchedule(LRSchedule):
    """Advance a PyTorch scheduler on the cadence it is written for."""

    def __init__(
        self,
        scheduler: LRScheduler,
        every: Cadence = "interval",
        monitor: str = "val_loss",
    ) -> None:
        """Bind the scheduler to when it advances and what it watches.

        Args:
            scheduler: Scheduler to advance, e.g. `CosineAnnealingLR`, whose
                lengths are counted in `every` units.
            every: "step" advances it after each optimizer step, "interval"
                after each interval: an epoch under `EpochSchedule`, an
                evaluation window under a step or token schedule.
            monitor: Recorded value a `ReduceLROnPlateau` steps on; its own
                `mode` says which direction improves. Other schedulers
                ignore it.

        Raises:
            ValueError: If a `ReduceLROnPlateau` is set to advance every
                step, where no recorded value exists yet.
        """
        if isinstance(scheduler, ReduceLROnPlateau) and every != "interval":
            raise ValueError(
                "ReduceLROnPlateau steps on recorded values, so it can only "
                'advance every "interval".'
            )

        self.scheduler = scheduler
        self.every = every
        self.monitor = monitor

    def after_step(self) -> None:
        """Advance the scheduler if it counts optimizer steps."""
        if self.every == "step":
            self.scheduler.step()

    def after_interval(self, values: Mapping[str, float]) -> None:
        """Advance the scheduler if it counts intervals.

        Args:
            values: What the interval recorded; a `ReduceLROnPlateau` steps
                on `values[monitor]`.

        Raises:
            ValueError: If a `ReduceLROnPlateau` monitors a value the
                interval did not record, such as "val_loss" without a
                validation loader.
        """
        if self.every != "interval":
            return

        if not isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step()
            return

        value = values.get(self.monitor)

        if value is None:
            raise ValueError(
                f"ReduceLROnPlateau monitors {self.monitor!r}, which the "
                "interval did not record; pass fit() a val_loader or monitor "
                "another value."
            )

        self.scheduler.step(value)

    def state_dict(self) -> dict[str, Any]:
        """Return the scheduler's state.

        Returns:
            dict[str, Any]: The wrapped scheduler's own state dict.
        """
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the scheduler's state.

        Args:
            state_dict: A mapping as returned by `state_dict`.
        """
        self.scheduler.load_state_dict(state_dict)


def as_lr_schedule(schedule: LRSchedule | LRScheduler) -> LRSchedule:
    """Accept a bare PyTorch scheduler where a schedule is expected.

    Args:
        schedule: A schedule, or a scheduler to advance every interval.

    Returns:
        LRSchedule: `schedule` itself, or the scheduler wrapped in a
            `TorchLRSchedule` with its defaults.
    """
    if isinstance(schedule, LRSchedule):
        return schedule

    return TorchLRSchedule(schedule)


class Warmup(LRSchedule):
    """Ramp the learning rate up linearly, then hand over to `then`."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        length: int,
        every: Cadence = "step",
        then: LRSchedule | LRScheduler | None = None,
    ) -> None:
        """Start the ramp and keep the schedule that follows it.

        Args:
            optimizer: Optimizer whose learning rate is ramped.
            length: How many `every` units the ramp spans; must be >= 1.
                Steps are optimizer steps, not batches, so gradient
                accumulation does not shorten the ramp.
            every: "step" ramps per optimizer step, "interval" per interval.
            then: Schedule that takes over once the ramp is done, on its own
                cadence; a bare PyTorch scheduler advances every interval.
                None leaves the rate flat afterwards.

        Raises:
            ValueError: If `length` is below 1.
        """
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")

        self.length = length
        self.every = every
        self.then = as_lr_schedule(then) if then is not None else None

        self.ramp = LinearLR(
            optimizer,
            start_factor=1.0 / length,
            end_factor=1.0,
            total_iters=length,
        )

    @property
    def in_warmup(self) -> bool:
        """Return whether the ramp is still running.

        Returns:
            bool: True until the ramp has advanced `length` times.
        """
        return self.ramp.last_epoch < self.length

    def after_step(self) -> None:
        """Advance the ramp per step, or pass the step on after it."""
        if not self.in_warmup:
            if self.then is not None:
                self.then.after_step()
            return

        if self.every == "step":
            self.ramp.step()

    def after_interval(self, values: Mapping[str, float]) -> None:
        """Advance the ramp per interval, or pass the interval on after it.

        Args:
            values: What the interval recorded, forwarded to `then`.
        """
        if not self.in_warmup:
            if self.then is not None:
                self.then.after_interval(values)
            return

        if self.every == "interval":
            self.ramp.step()

    def state_dict(self) -> dict[str, Any]:
        """Return the ramp's and the following schedule's state.

        Returns:
            dict[str, Any]: State under "warmup" and "scheduler"; the latter
                is None without a following schedule.
        """
        return {
            "warmup": self.ramp.state_dict(),
            "scheduler": self.then.state_dict() if self.then is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the ramp's and the following schedule's state.

        The learning rate itself travels with the optimizer's state, so it
        is restored by loading that.

        Args:
            state_dict: A mapping as returned by `state_dict`.
        """
        self.ramp.load_state_dict(state_dict["warmup"])

        then_state = state_dict.get("scheduler")
        if self.then is not None and then_state is not None:
            self.then.load_state_dict(then_state)
