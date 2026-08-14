"""Loss aggregation strategies for use with Trainer."""

from abc import ABC, abstractmethod

import torch


class LossTracker(ABC):
    """Base interface for aggregating per-batch loss over an epoch.

    Subclasses accumulate state from each batch via `update` and expose
    the aggregated result via `compute`. Useful when a plain per-batch
    average (mean of per-batch losses) doesn't reflect the quantity you
    actually care about, e.g. per-token loss on padded sequences.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state, e.g. at the start of an epoch."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        loss: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        extras: list[torch.Tensor],
        predictions: torch.Tensor,
        train: bool,
    ) -> None:
        """Accumulate state from one batch.

        Args:
            loss: Batch loss, as returned by the trainer's loss function.
            inputs: Batch inputs, already moved to the training device.
            targets: Batch targets, already moved to the training device.
            extras: Any batch elements beyond inputs/targets, already moved
                to the training device.
            predictions: Model predictions for this batch.
            train: Whether this batch was part of a training pass.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        """Return the aggregated loss over all batches seen since `reset`."""
        raise NotImplementedError

    def batch_weight(self, _targets: torch.Tensor) -> float:
        """Return one batch's contribution to `compute`'s denominator.

        Defaults to 1.0, matching an aggregate that divides by the batch
        count. Subclasses dividing by something else should override this.

        Args:
            _targets: Batch targets, already moved to the training device.

        Returns:
            The weight this batch carries when averaging across batches.
        """
        return 1.0


class MeanLossTracker(LossTracker):
    """Averages loss per batch, i.e. the mean of per-batch loss values."""

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
        _inputs: torch.Tensor,
        _targets: torch.Tensor,
        _extras: list[torch.Tensor],
        _predictions: torch.Tensor,
        _train: bool,
    ) -> None:
        """Accumulate one batch's loss.

        Args:
            loss: Batch loss, as returned by the trainer's loss function.
            _inputs: Unused.
            _targets: Unused.
            _extras: Unused.
            _predictions: Unused.
            _train: Unused.
        """
        self._total_loss += loss.item()
        self._n_batches += 1

    def compute(self) -> float:
        """Return the mean loss over all batches seen since `reset`."""
        return self._total_loss / max(self._n_batches, 1)


class PerTokenLossTracker(LossTracker):
    """Averages loss per non-padding token, rather than per batch.

    Useful for sequence models where batches have a varying number of
    non-padding target tokens: a per-batch average would over-weight
    batches with heavier padding relative to their actual token count.
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
        _inputs: torch.Tensor,
        targets: torch.Tensor,
        _extras: list[torch.Tensor],
        _predictions: torch.Tensor,
        _train: bool,
    ) -> None:
        """Accumulate one batch's loss and non-padding token count.

        Args:
            loss: Batch loss, as returned by the trainer's loss function.
            _inputs: Unused.
            targets: Batch targets, used to count non-padding tokens.
            _extras: Unused.
            _predictions: Unused.
            _train: Unused.
        """
        self._total_loss += loss.item()
        self._total_tokens += int((targets != self.pad_id).sum().item())

    def compute(self) -> float:
        """Return the mean loss per non-padding token seen since `reset`."""
        return self._total_loss / max(self._total_tokens, 1)

    def batch_weight(self, targets: torch.Tensor) -> float:
        """Return the number of non-padding tokens in `targets`.

        Args:
            targets: Batch targets, already moved to the training device.

        Returns:
            The weight this batch carries when averaging across batches.
        """
        return float((targets != self.pad_id).sum().item())
