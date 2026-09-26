"""Trainer variant feeding the decoder the input its batch carries."""

from dataclasses import dataclass

import torch

from dl_roadmap.engine.trainer.batch import Batch, BatchParts
from dl_roadmap.engine.trainer.context import StepContext
from dl_roadmap.engine.trainer.trainer import Trainer


@dataclass(frozen=True)
class TeacherForcingParts(BatchParts):
    """A batch's parts with the tensor the decoder is fed named.

    Attributes:
        decoder_input: What the decoder reads at each position, one step
            behind `targets`; None when absent from the batch.
    """

    decoder_input: torch.Tensor | None


class TeacherForcingTrainer[BatchT: Batch](Trainer[BatchT]):
    """Trainer that feeds the decoder input in both training and validation."""

    def _split(self, batch: BatchT) -> TeacherForcingParts:
        """Name a batch's tensors, the decoder input among them.

        Args:
            batch: Source, target and decoder input tensors in that order,
                already on `self.device`; only source is required, and
                anything past the decoder input is ignored.

        Returns:
            TeacherForcingParts: Named tensors, with absent targets and
                decoder input set to None.
        """
        inputs, *rest = batch

        return TeacherForcingParts(
            inputs=inputs,
            targets=rest[0] if rest else None,
            decoder_input=rest[1] if len(rest) > 1 else None,
        )

    def _forward(self, batch: BatchT, ctx: StepContext) -> torch.Tensor:
        """Use teacher forcing unless predicting from the source alone.

        Args:
            batch: Source, target and decoder input tensors, in that order,
                already on `self.device`; prediction needs only the source.
            ctx: Run phase; prediction calls `model(source)` without targets.

        Returns:
            torch.Tensor: Model predictions, passed to `self.loss_fn`
                alongside the targets.
        """
        parts = self._split(batch)

        if ctx.phase == "predict":
            return self.model(parts.inputs)  # type: ignore[no-any-return]

        return self.model(parts.inputs, parts.decoder_input)  # type: ignore[no-any-return]
