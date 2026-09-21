"""Trainer variant feeding the decoder its own input tensor."""

import torch

from dl_roadmap.engine.trainer.base import BaseTrainer, Batch
from dl_roadmap.engine.trainer.context import StepContext

DECODER_INPUT = 2


class TeacherForcingTrainer[BatchT: Batch](BaseTrainer[BatchT]):
    """Trainer that feeds the decoder input in both training and validation."""

    def _forward(self, batch: BatchT, _ctx: StepContext) -> torch.Tensor:
        """Calls the model with the decoder input carried by the batch.

        Args:
            batch: Source, target and decoder input tensors, in that order,
                already on `self.device`.
            _ctx: Phase, epoch and step the batch belongs to; unused here.

        Returns:
            Model predictions, passed to `self.loss_fn` alongside targets.

        Raises:
            ValueError: If the batch carries no decoder input.
        """
        if len(batch) <= DECODER_INPUT:
            raise ValueError(
                "TeacherForcingTrainer needs a (source, target, decoder input) "
                f"batch; got {len(batch)} tensor(s)."
            )

        return self.model(batch[0], batch[DECODER_INPUT])  # type: ignore[no-any-return]
