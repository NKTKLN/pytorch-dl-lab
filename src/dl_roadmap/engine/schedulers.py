"""Learning rate schedules that PyTorch cannot compose on its own."""

from typing import Any

import torch
from torch.optim.lr_scheduler import LinearLR, LRScheduler, ReduceLROnPlateau


def step_scheduler(scheduler: LRScheduler, val_loss: float | None = None) -> None:
    """Advances a scheduler, passing `val_loss` only to the ones that need it.

    Args:
        scheduler: The scheduler to advance.
        val_loss: Current epoch's validation loss. Required when
            `scheduler` is a `ReduceLROnPlateau`, which steps on this
            metric instead of unconditionally.

    Raises:
        ValueError: If `scheduler` is a `ReduceLROnPlateau` but no
            `val_loss` was provided.
    """
    if isinstance(scheduler, ReduceLROnPlateau):
        if val_loss is None:
            raise ValueError(
                "ReduceLROnPlateau requires a validation loss; "
                "call fit() with a val_loader."
            )

        scheduler.step(float(val_loss))
    else:
        scheduler.step()


class WarmupScheduler:
    """Linear LR warmup by optimizer step, then hands over to `scheduler`.

    Attributes:
        optimizer: The optimizer whose learning rate is scheduled.
        warmup_steps: Number of optimizer steps the ramp spans.
        warmup: The `LinearLR` doing the ramp.
        scheduler: The scheduler that takes over afterwards, if any.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        scheduler: LRScheduler | None = None,
    ) -> None:
        """Initializes the warmup ramp and stores the scheduler after it.

        Args:
            optimizer: The optimizer to schedule.
            warmup_steps: Number of optimizer steps to ramp over, counted
                in optimizer steps rather than batches, so gradient
                accumulation does not silently shorten the ramp.
            scheduler: Scheduler to advance once warmup is done. None runs
                warmup alone and leaves the rate flat afterwards.

        Raises:
            ValueError: If `warmup_steps` is below 1.
        """
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler

        self.warmup = LinearLR(
            optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

    @property
    def in_warmup(self) -> bool:
        """Returns whether the ramp is still running."""
        return self.warmup.last_epoch < self.warmup_steps

    def step_batch(self) -> None:
        """Advances the ramp by one optimizer step; a no-op once it is done."""
        if self.in_warmup:
            self.warmup.step()

    def step_epoch(self, val_loss: float | None = None) -> None:
        """Advances the wrapped scheduler, once warmup has handed over.

        Args:
            val_loss: Current epoch's validation loss, forwarded to the
                wrapped scheduler if it needs one.
        """
        if self.scheduler is None or self.in_warmup:
            return

        step_scheduler(self.scheduler, val_loss)

    def get_last_lr(self) -> list[float]:
        """Returns the current learning rate of every param group."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Returns the ramp's and the wrapped scheduler's state."""
        return {
            "warmup": self.warmup.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the ramp's and the wrapped scheduler's state.

        Args:
            state_dict: A dict as returned by `state_dict`.
        """
        self.warmup.load_state_dict(state_dict["warmup"])

        scheduler_state = state_dict.get("scheduler")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self._apply_last_lr()

    def _apply_last_lr(self) -> None:
        """Writes the restored schedule's rate back onto the param groups."""
        active: Any = self.warmup
        if not self.in_warmup and self.scheduler is not None:
            active = self.scheduler

        last_lr = getattr(active, "_last_lr", None)
        if last_lr is None:
            return

        for group, lr in zip(self.optimizer.param_groups, last_lr):
            group["lr"] = float(lr)
