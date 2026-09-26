"""When a run stops, and which interval supplied its best weights."""

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from loguru import logger
from torch import nn

from dl_roadmap.engine.trainer.report import IntervalReport

Mode = Literal["min", "max"]
"""Which direction a score must move to improve."""

Combine = Literal["all", "any"]
"""How several strategies agree that a run should stop."""


class EarlyStopping(ABC):
    """Track the best interval and whether a run should stop."""

    def __init__(self, restore_best_weights: bool = False) -> None:
        """Start without a best interval or a reason to stop.

        Args:
            restore_best_weights: Keep a copy of the model's weights from the
                best interval, for `restore` to put back when the run ends.
        """
        self.restore_best_weights = restore_best_weights

        self.best_interval: int | None = None
        self.is_best: bool = False
        self.should_stop: bool = False

        self._best_state: dict[str, torch.Tensor] | None = None

    def update(self, report: IntervalReport) -> None:
        """Record whether this interval is the best or ends the run.

        Args:
            report: Finished interval's losses, metrics and place in the run;
                its model is copied when the interval is the best so far and
                `restore_best_weights` is set.
        """
        self._decide(report)

        if self.is_best and self.restore_best_weights:
            self._best_state = {
                name: tensor.detach().to("cpu", copy=True)
                for name, tensor in report.store.model.state_dict().items()
            }

    def restore(self, model: nn.Module) -> None:
        """Put the best interval's weights back into `model`.

        Args:
            model: Model the run trained. Left unchanged when no best weights
                were kept.
        """
        if self._best_state is None:
            return

        model.load_state_dict(self._best_state)
        logger.debug(f"Restored best model weights from interval {self.best_interval}")

    @abstractmethod
    def _decide(self, report: IntervalReport) -> None:
        """Set `is_best`, `best_interval` and `should_stop` for an interval.

        Args:
            report: Finished interval's losses, metrics and place in the run.
        """
        raise NotImplementedError


class ThresholdEarlyStopping(EarlyStopping):
    """Stop after `patience` scored intervals without an improvement."""

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: Mode = "min",
        *,
        restore_best_weights: bool = False,
    ) -> None:
        """Choose how much improvement to require and how long to wait.

        Args:
            patience: Scored intervals without improvement needed to stop;
                must be >= 1. An improvement resets the count.
            min_delta: Change from the best score must strictly exceed this
                value to count as an improvement.
            mode: "min" if a lower score is better, "max" if a higher score
                is better.
            restore_best_weights: Keep the best interval's weights for
                `restore`.

        Raises:
            ValueError: If `patience` is less than 1.
        """
        super().__init__(restore_best_weights)

        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score: float | None = None
        self._intervals_without_improvement = 0

    @abstractmethod
    def _score(self, report: IntervalReport) -> float | None:
        """Return the value this strategy tracks for one interval.

        Args:
            report: What the interval recorded.

        Returns:
            float | None: The score to track, or None when it cannot be
                computed for this interval.
        """
        raise NotImplementedError

    def _is_improved(self, score: float) -> bool:
        """Check whether the score improves by more than `min_delta`.

        Args:
            score: Score of the interval just finished.

        Returns:
            bool: False for a NaN or infinite score, which a diverged run
                reports; otherwise True for the first score, and afterwards
                only for one that clears the best by more than `min_delta`.
        """
        if not math.isfinite(score):
            return False

        if self.best_score is None:
            return True

        if self.mode == "min":
            return score < self.best_score - self.min_delta

        return score > self.best_score + self.min_delta

    def _decide(self, report: IntervalReport) -> None:
        """Record an improvement or count another interval toward stopping.

        Args:
            report: Interval to score. A missing score clears `is_best`
                but leaves the best interval, counter and stop flag unchanged.
        """
        score = self._score(report)
        self.is_best = False

        if score is None:
            return

        interval = report.ctx.interval

        if self._is_improved(score):
            logger.debug(
                f"{type(self).__name__}: improved at interval {interval}: "
                f"score={score:.4g} (best was {self.best_score})"
            )
            self.best_score = score
            self.best_interval = interval
            self.is_best = True
            self._intervals_without_improvement = 0
        else:
            self._intervals_without_improvement += 1

        self.should_stop = self._intervals_without_improvement >= self.patience

        if self.should_stop:
            logger.info(
                f"{type(self).__name__}: should stop at interval {interval}: "
                f"no improvement for {self._intervals_without_improvement} intervals"
            )


class ValLossEarlyStopping(ThresholdEarlyStopping):
    """Stop when validation loss stops decreasing."""

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        *,
        restore_best_weights: bool = False,
    ) -> None:
        """Track validation loss, counting a decrease as an improvement.

        Args:
            patience: Scored intervals without improvement needed to stop;
                must be >= 1. An improvement resets the count.
            min_delta: Decrease from the best validation loss must strictly
                exceed this value to count as an improvement.
            restore_best_weights: Keep the best interval's weights for
                `restore`.

        Raises:
            ValueError: If `patience` is less than 1.
        """
        super().__init__(
            patience, min_delta, mode="min", restore_best_weights=restore_best_weights
        )

    def _score(self, report: IntervalReport) -> float | None:
        """Return the interval's validation loss.

        Args:
            report: What the interval recorded.

        Returns:
            float | None: The validation loss, or None without validation.
        """
        return report.val_loss


class MetricEarlyStopping(ThresholdEarlyStopping):
    """Stop when a metric recorded by the run stops improving."""

    def __init__(
        self,
        key: str,
        patience: int,
        min_delta: float = 0.0,
        mode: Mode = "max",
        *,
        restore_best_weights: bool = False,
    ) -> None:
        """Track one recorded metric in the chosen direction.

        Args:
            key: Key to track in `report.values`, e.g. "val_rouge_mean".
            patience: Scored intervals without improvement needed to stop;
                must be >= 1. An improvement resets the count.
            min_delta: Change from the best metric value must strictly
                exceed this value to count as an improvement.
            mode: "max" if a higher value is better, "min" if lower is.
            restore_best_weights: Keep the best interval's weights for
                `restore`.

        Raises:
            ValueError: If `patience` is less than 1.
        """
        super().__init__(
            patience, min_delta, mode=mode, restore_best_weights=restore_best_weights
        )

        self.key = key

    def _score(self, report: IntervalReport) -> float | None:
        """Return what this interval recorded under `key`.

        Args:
            report: What the interval recorded.

        Returns:
            float | None: This interval's value, or None if the key is absent.
        """
        return report.values.get(self.key)


class GeneralizationGapEarlyStopping(ThresholdEarlyStopping):
    """Stop when the val/train loss gap stops shrinking."""

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        *,
        restore_best_weights: bool = False,
    ) -> None:
        """Track the signed loss gap, counting a decrease as an improvement.

        Args:
            patience: Scored intervals without improvement needed to stop;
                must be >= 1. An improvement resets the count.
            min_delta: Decrease from the best signed val/train gap must
                strictly exceed this value to count as an improvement.
            restore_best_weights: Keep the best interval's weights for
                `restore`.

        Raises:
            ValueError: If `patience` is less than 1.
        """
        super().__init__(
            patience, min_delta, mode="min", restore_best_weights=restore_best_weights
        )

    def _score(self, report: IntervalReport) -> float | None:
        """Return how far validation loss sits above training loss.

        Args:
            report: What the interval recorded.

        Returns:
            float | None: `val_loss - train_loss`, or None without validation.
        """
        if report.val_loss is None:
            return None

        return report.val_loss - report.train_loss


class GapThresholdEarlyStopping(EarlyStopping):
    """Stop once the val/train loss gap stays above a fixed threshold."""

    def __init__(self, patience: int, threshold: float) -> None:
        """Choose the largest allowed loss gap and how long it may persist.

        Args:
            patience: Number of consecutive intervals the gap must exceed
                `threshold` before `should_stop` is set.
            threshold: Largest tolerated `val_loss - train_loss`.

        Raises:
            ValueError: If `patience` is less than 1.
        """
        super().__init__()

        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")

        self.patience = patience
        self.threshold = threshold

        self.gap: float | None = None
        self._intervals_over_threshold = 0

    def _decide(self, report: IntervalReport) -> None:
        """Count a gap above the threshold, or reset the count below it.

        Args:
            report: Losses to compare. A gap at or below the threshold
                resets the count; missing validation leaves state untouched.
        """
        if report.val_loss is None:
            return

        self.gap = report.val_loss - report.train_loss

        if self.gap > self.threshold:
            self._intervals_over_threshold += 1
        else:
            self._intervals_over_threshold = 0

        self.should_stop = self._intervals_over_threshold >= self.patience

        if self.should_stop:
            logger.info(
                f"{type(self).__name__}: should stop at interval "
                f"{report.ctx.interval}: gap={self.gap:.4g} above "
                f"threshold={self.threshold} for "
                f"{self._intervals_over_threshold} intervals"
            )


class CombinedEarlyStopping(EarlyStopping):
    """Combine several early stopping strategies into one."""

    def __init__(
        self,
        strategies: list[EarlyStopping],
        combine: Combine = "any",
        *,
        restore_best_weights: bool = False,
    ) -> None:
        """Choose the strategies and whether any or all must request a stop.

        Args:
            strategies: Strategies to update together; must not be empty.
            combine: "any" stops as soon as one strategy wants to stop;
                "all" waits until every strategy wants to stop. It does not
                apply to `is_best`: any improvement marks this interval as
                best, even when another strategy's score gets worse.
            restore_best_weights: Keep the best interval's weights for
                `restore`. Set it here, not on the wrapped strategies.

        Raises:
            ValueError: If `strategies` is empty, or if a wrapped strategy
                sets `restore_best_weights` itself.
        """
        super().__init__(restore_best_weights)

        if not strategies:
            raise ValueError("strategies must not be empty.")

        if any(strategy.restore_best_weights for strategy in strategies):
            raise ValueError(
                "Set restore_best_weights on CombinedEarlyStopping, "
                "not on the strategies it wraps."
            )

        self.strategies = strategies
        self.combine = combine

    def _decide(self, report: IntervalReport) -> None:
        """Update every wrapped strategy and aggregate their flags.

        Args:
            report: What the interval recorded, passed on unchanged.
        """
        for strategy in self.strategies:
            strategy.update(report)

        stop_flags = [strategy.should_stop for strategy in self.strategies]
        self.should_stop = any(stop_flags) if self.combine == "any" else all(stop_flags)

        if self.should_stop:
            logger.info(
                f"CombinedEarlyStopping ({self.combine}): should stop at "
                f"interval {report.ctx.interval}"
            )

        self.is_best = any(strategy.is_best for strategy in self.strategies)

        if self.is_best:
            self.best_interval = report.ctx.interval
