"""What a loader yields, and the roles a trainer reads it in."""

from dataclasses import dataclass

import torch

Batch = tuple[torch.Tensor, ...]
"""One batch: inputs, then targets, then any extra tensors the model needs."""

PairBatch = tuple[torch.Tensor, torch.Tensor]
"""The common case: inputs and targets, nothing else."""


@dataclass(frozen=True)
class BatchParts:
    """The roles the trainer reads a batch's tensors in.

    Attributes:
        inputs: What the model is called with, and what the metrics score.
        targets: What `loss_fn` and the metrics compare against; None for a
            batch of inputs alone, as `predict` allows. A variant that reads
            more of the batch names it on a subclass, the way
            `TeacherForcingParts` names the decoder input.
    """

    inputs: torch.Tensor
    targets: torch.Tensor | None

    def require_targets(self) -> torch.Tensor:
        """Return the targets of a batch that has to carry them.

        Returns:
            torch.Tensor: The batch's targets.

        Raises:
            ValueError: If the batch holds inputs alone.
        """
        if self.targets is None:
            raise ValueError("This batch carries no targets.")

        return self.targets
