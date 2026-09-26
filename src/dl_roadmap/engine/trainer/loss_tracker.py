"""Loss aggregation strategies for use with Trainer."""

from abc import ABC, abstractmethod

import torch

from dl_roadmap.engine.trainer.batch import BatchParts
from dl_roadmap.engine.trainer.context import StepContext


class LossTracker(ABC):
    """Base interface for aggregating per-batch loss over one pass."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state before a training or validation pass."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        loss: torch.Tensor,
        parts: BatchParts,
        predictions: torch.Tensor,
        ctx: StepContext,
    ) -> None:
        """Accumulate state from one batch.

        Args:
            loss: Batch loss from the trainer's loss function, detached and
                cast to float32.
            parts: Batch tensors by role, already on the training device;
                targets are present.
            predictions: Model predictions for this batch, detached.
            ctx: Phase, interval and step this batch belongs to.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        """Return the loss aggregated over the batches seen since `reset`.

        Returns:
            float: The aggregated loss; 0.0 before any batch.
        """
        raise NotImplementedError

    def batch_weight(self, _parts: BatchParts) -> float:
        """Return one batch's contribution to `compute`'s denominator.

        The optimization engine also normalizes accumulated gradients by the
        summed weights when `grad_normalizer` is "loss_weights".

        Args:
            _parts: Batch tensors by role, already on the training device.

        Returns:
            float: The weight this batch carries; 1.0 counts every batch
                equally.
        """
        return 1.0


class MeanLossTracker(LossTracker):
    """Average the per-batch losses, each batch counting once.

    Fits a mean-reduced loss, such as `nn.CrossEntropyLoss()` by default.
    """

    def __init__(self) -> None:
        """Initialize the tracker with empty state."""
        self._total_loss = 0.0
        self._n_batches = 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._total_loss = 0.0
        self._n_batches = 0

    def update(
        self,
        loss: torch.Tensor,
        _parts: BatchParts,
        _predictions: torch.Tensor,
        _ctx: StepContext,
    ) -> None:
        """Accumulate one batch's loss.

        Args:
            loss: Batch loss, as returned by the trainer's loss function.
            _parts: Unused.
            _predictions: Unused.
            _ctx: Unused.
        """
        self._total_loss += loss.item()
        self._n_batches += 1

    def compute(self) -> float:
        """Return the mean of the batch losses seen since `reset`.

        Returns:
            float: Summed batch losses over the batch count; 0.0 before any
                batch.
        """
        return self._total_loss / max(self._n_batches, 1)


class PerTokenLossTracker(LossTracker):
    """Average the loss per non-padding target token, rather than per batch.

    Fits a loss summed over tokens, as `make_token_loss` builds; a
    mean-reduced loss would be divided by the token count twice.
    """

    def __init__(self, pad_id: int) -> None:
        """Initialize the tracker with empty state.

        Args:
            pad_id: Token id used for padding, excluded from the token count.
        """
        self.pad_id = pad_id
        self._total_loss = 0.0
        self._total_tokens = 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._total_loss = 0.0
        self._total_tokens = 0

    def update(
        self,
        loss: torch.Tensor,
        parts: BatchParts,
        _predictions: torch.Tensor,
        _ctx: StepContext,
    ) -> None:
        """Accumulate one batch's loss and non-padding token count.

        Args:
            loss: Batch loss, as returned by the trainer's loss function.
            parts: Batch tensors by role; the targets are used to count
                non-padding tokens.
            _predictions: Unused.
            _ctx: Unused.
        """
        targets = parts.require_targets()
        self._total_loss += loss.item()
        self._total_tokens += int((targets != self.pad_id).sum().item())

    def compute(self) -> float:
        """Return the mean loss per non-padding token seen since `reset`.

        Returns:
            float: Summed batch losses over the non-padding token count; 0.0
                before any batch.
        """
        return self._total_loss / max(self._total_tokens, 1)

    def batch_weight(self, parts: BatchParts) -> float:
        """Return the number of non-padding tokens in the batch's targets.

        Args:
            parts: Batch tensors by role, already on the training device.

        Returns:
            float: The batch's non-padding target token count.
        """
        return float((parts.require_targets() != self.pad_id).sum().item())
