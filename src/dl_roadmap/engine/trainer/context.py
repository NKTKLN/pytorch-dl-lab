"""Context describing where in training a forward pass happens."""

from dataclasses import dataclass
from typing import Literal

Phase = Literal["train", "val", "predict"]
"""Pass a batch belongs to."""

Unit = Literal["batch", "token", "step"]
"""What a pass counts its work in."""

AxisUnit = Literal["epoch", "batch", "token", "step"]
"""What a run counts its intervals in."""


@dataclass(frozen=True)
class StepContext:
    """Where in the run the current batch is being processed.

    Attributes:
        phase: Pass the batch belongs to. "val" and "predict" both run without
            gradients, but only "val" has targets to compare against.
        interval: Interval the pass belongs to, counted from 1; 0 outside
            of `fit`. Under an epoch schedule it is the epoch number.
        intervals: Intervals the run spans; 0 outside of `fit`.
        unit: What `interval` counts, which an epoch schedule leaves at
            "epoch" and a step or token schedule sets to its own.
        step: Optimizer steps performed so far in the run.
    """

    phase: Phase
    interval: int = 0
    intervals: int = 0
    unit: AxisUnit = "epoch"
    step: int = 0

    @property
    def is_training(self) -> bool:
        """Return whether this pass updates the model.

        Returns:
            bool: True for the "train" phase.
        """
        return self.phase == "train"
