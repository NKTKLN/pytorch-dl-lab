"""Context describing where in training a forward pass happens."""

from dataclasses import dataclass
from typing import Literal

Phase = Literal["train", "val", "predict"]


@dataclass(frozen=True)
class StepContext:
    """Where in the run the current batch is being processed.

    Attributes:
        phase: Pass the batch belongs to. "val" and "predict" both run without
            gradients, but only "val" has targets to compare against.
        epoch: Epoch number, counted from 1; 0 outside of `fit`.
        epochs: Total epochs the run was configured for; 0 outside of `fit`.
        step: Optimizer steps performed so far in the run.
    """

    phase: Phase
    epoch: int = 0
    epochs: int = 0
    step: int = 0

    @property
    def is_training(self) -> bool:
        """Return whether this pass updates the model."""
        return self.phase == "train"

    @property
    def progress(self) -> float:
        """Return how far the run has got, as a fraction of its epochs.

        Returns:
            float: Value in [0, 1]; 0 when the total epoch count is unknown.
        """
        if self.epochs < 1:
            return 0.0

        return min(1.0, (self.epoch - 1) / self.epochs)
