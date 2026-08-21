"""Early stopping strategies for use with Trainer."""

from abc import ABC, abstractmethod
from typing import Literal

from loguru import logger

Mode = Literal["min", "max"]
Combine = Literal["all", "any"]


class EarlyStopping(ABC):
    """Base interface for early stopping strategies.

    Attributes:
        best_epoch: The epoch with the best score seen so far. None until
            an improvement is observed.
        is_best: Whether the most recent `update` call was an improvement.
        should_stop: Whether training should stop after the most recent
            `update` call.
    """

    def __init__(self) -> None:
        """Initialize stopping state; subclasses must call this via super()."""
        self.best_epoch: int | None = None
        self.is_best: bool = False
        self.should_stop: bool = False

    @abstractmethod
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        history: dict[str, list[float]],
    ) -> None:
        """Update stopping state with the results of one epoch.

        Args:
            epoch: The 1-indexed epoch that just finished.
            train_loss: Average training loss for this epoch.
            val_loss: Average validation loss for this epoch, or None if
                no validation loader was used.
            history: Per-epoch "train_loss"/"val_loss" values recorded so far.
        """
        raise NotImplementedError


class ThresholdEarlyStopping(EarlyStopping):
    """Stops after `patience` consecutive epochs without a scored improvement."""

    def __init__(
        self, patience: int, min_delta: float = 0.0, mode: Mode = "min"
    ) -> None:
        """Initialize the strategy.

        Args:
            patience: Number of consecutive non-improving epochs to tolerate
                before `should_stop` is set.
            min_delta: Minimum change in score (vs. the best seen so far) to
                count as an improvement.
            mode: "min" if a lower score is better, "max" if a higher score
                is better.

        Raises:
            ValueError: If `patience` is less than 1.
        """
        super().__init__()

        if patience < 1:
            raise ValueError("patience must be >= 1")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score: float | None = None
        self._epochs_without_improvement = 0

    @abstractmethod
    def _score(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        history: dict[str, list[float]],
    ) -> float | None:
        """Compute the metric to track for this epoch.

        Args:
            epoch: The 1-indexed epoch that just finished.
            train_loss: Average training loss for this epoch.
            val_loss: Average validation loss for this epoch, or None if
                no validation loader was used.
            history: Per-epoch "train_loss"/"val_loss" values recorded so far.

        Returns:
            The score to track, or None if it can't be computed yet (e.g.
            no `val_loss` available), in which case `update` is a no-op.
        """
        raise NotImplementedError

    def _is_improved(self, score: float) -> bool:
        """Return whether `score` beats `best_score` by at least `min_delta`."""
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        history: dict[str, list[float]],
    ) -> None:
        """Update stopping state with the results of one epoch.

        Args:
            epoch: The 1-indexed epoch that just finished.
            train_loss: Average training loss for this epoch.
            val_loss: Average validation loss for this epoch, or None if
                no validation loader was used.
            history: Per-epoch "train_loss"/"val_loss" values recorded so far.
        """
        score = self._score(epoch, train_loss, val_loss, history)
        self.is_best = False

        if score is None:
            return

        if self._is_improved(score):
            logger.debug(
                f"{type(self).__name__}: improved at epoch {epoch}: "
                f"score={score:.4g} (best was {self.best_score})"
            )
            self.best_score = score
            self.best_epoch = epoch
            self.is_best = True
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        self.should_stop = self._epochs_without_improvement >= self.patience
        if self.should_stop:
            logger.info(
                f"{type(self).__name__}: should stop at epoch {epoch}: "
                f"no improvement for {self._epochs_without_improvement} epochs"
            )


class ValLossEarlyStopping(ThresholdEarlyStopping):
    """Stops when validation loss stops improving. The common default."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """Initialize the strategy.

        Args:
            patience: Number of consecutive non-improving epochs to tolerate
                before `should_stop` is set.
            min_delta: Minimum decrease in val_loss (vs. the best seen so
                far) to count as an improvement.

        Raises:
            ValueError: If `patience` is less than 1.
        """
        super().__init__(patience, min_delta, mode="min")

    def _score(
        self,
        _epoch: int,
        _train_loss: float,
        val_loss: float | None,
        _history: dict[str, list[float]],
    ) -> float | None:
        """Return `val_loss`, or None if no validation loader was used."""
        return val_loss


class GeneralizationGapEarlyStopping(ThresholdEarlyStopping):
    """Stops when the val/train loss gap stops shrinking, i.e. overfitting."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """Initialize the strategy.

        Args:
            patience: Number of consecutive non-improving epochs to tolerate
                before `should_stop` is set.
            min_delta: Minimum decrease in the val/train gap (vs. the best
                seen so far) to count as an improvement.

        Raises:
            ValueError: If `patience` is less than 1.
        """
        super().__init__(patience, min_delta, mode="min")

    def _score(
        self,
        _epoch: int,
        train_loss: float,
        val_loss: float | None,
        _history: dict[str, list[float]],
    ) -> float | None:
        """Return `abs(val_loss - train_loss)`, or None if `val_loss` is unset."""
        if val_loss is None:
            return None
        return abs(val_loss - train_loss)


class GapThresholdEarlyStopping(EarlyStopping):
    """Stops once the val/train loss gap exceeds a fixed threshold.

    Attributes:
        gap: The `abs(val_loss - train_loss)` of the most recent `update`
            call. None until an epoch with a `val_loss` is seen.
    """

    def __init__(self, patience: int, threshold: float) -> None:
        """Initialize the strategy.

        Args:
            patience: Number of consecutive epochs the gap must exceed
                `threshold` before `should_stop` is set.
            threshold: Maximum tolerated `abs(val_loss - train_loss)`.

        Raises:
            ValueError: If `patience` is less than 1.
        """
        super().__init__()

        if patience < 1:
            raise ValueError("patience must be >= 1")

        self.patience = patience
        self.threshold = threshold

        self.gap: float | None = None
        self._epochs_over_threshold = 0

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        _history: dict[str, list[float]],
    ) -> None:
        """Update stopping state with the results of one epoch.

        Args:
            epoch: The 1-indexed epoch that just finished.
            train_loss: Average training loss for this epoch.
            val_loss: Average validation loss for this epoch, or None if
                no validation loader was used.
            _history: Per-epoch "train_loss"/"val_loss" values recorded so
                far. Unused.
        """
        if val_loss is None:
            return

        self.gap = abs(val_loss - train_loss)
        if self.gap > self.threshold:
            self._epochs_over_threshold += 1
        else:
            self._epochs_over_threshold = 0

        self.should_stop = self._epochs_over_threshold >= self.patience
        if self.should_stop:
            logger.info(
                f"{type(self).__name__}: should stop at epoch {epoch}: "
                f"gap={self.gap:.4g} above threshold={self.threshold} for "
                f"{self._epochs_over_threshold} epochs"
            )


class CombinedEarlyStopping(EarlyStopping):
    """Combines multiple early stopping strategies into one."""

    def __init__(
        self, strategies: list[EarlyStopping], combine: Combine = "any"
    ) -> None:
        """Initialize the strategy.

        Args:
            strategies: Early stopping strategies to combine.
            combine: "any" stops as soon as one strategy wants to stop;
                "all" waits until every strategy wants to stop.

        Raises:
            ValueError: If `strategies` is empty.
        """
        super().__init__()

        if not strategies:
            raise ValueError("strategies must not be empty")

        self.strategies = strategies
        self.combine = combine

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        history: dict[str, list[float]],
    ) -> None:
        """Update every wrapped strategy and aggregate their flags.

        Args:
            epoch: The 1-indexed epoch that just finished.
            train_loss: Average training loss for this epoch.
            val_loss: Average validation loss for this epoch, or None if
                no validation loader was used.
            history: Per-epoch "train_loss"/"val_loss" values recorded so far.
        """
        for strategy in self.strategies:
            strategy.update(epoch, train_loss, val_loss, history)

        stop_flags = [s.should_stop for s in self.strategies]
        self.should_stop = any(stop_flags) if self.combine == "any" else all(stop_flags)
        if self.should_stop:
            logger.info(
                f"CombinedEarlyStopping ({self.combine}): should stop at epoch {epoch}"
            )

        best_flags = [s.is_best for s in self.strategies]
        self.is_best = any(best_flags) if self.combine == "any" else all(best_flags)
        if self.is_best:
            self.best_epoch = epoch
