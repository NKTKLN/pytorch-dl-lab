"""Keeps the epoch loss and metric values in one place."""

import torch

from dl_roadmap.engine.trainer.context import StepContext
from dl_roadmap.engine.trainer.loss_tracker import LossTracker, MeanLossTracker
from dl_roadmap.engine.trainer.online_metrics import Metric, flatten_metric

History = dict[str, list[float]]


class MetricsManager:
    """Aggregate the loss and every configured metric over an epoch."""

    def __init__(
        self,
        loss_tracker: LossTracker | None = None,
        metrics: dict[str, Metric] | None = None,
    ) -> None:
        """Bind the loss tracker and metrics updated on every batch.

        Args:
            loss_tracker: Aggregate batch losses into an epoch loss.
                Defaults to `MeanLossTracker`, which averages batch losses.
            metrics: Named metrics updated during training and validation.
                Results are stored in the history under `train_<name>` and
                `val_<name>`.
        """
        self.loss_tracker = loss_tracker or MeanLossTracker()
        self.metrics = metrics or {}

    def reset(self) -> None:
        """Clear the loss tracker and every metric before a pass."""
        self.loss_tracker.reset()

        for metric in self.metrics.values():
            metric.reset()

    def update(
        self,
        loss: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        extras: list[torch.Tensor],
        predictions: torch.Tensor,
        ctx: StepContext,
    ) -> None:
        """Feed one batch to the loss tracker and every metric.

        Args:
            loss: Detached scalar loss for the batch.
            inputs: Batch inputs.
            targets: Batch targets.
            extras: Additional batch tensors.
            predictions: Detached model predictions.
            ctx: Phase, epoch and step this batch belongs to.
        """
        self.loss_tracker.update(loss, inputs, targets, extras, predictions, ctx)

        for metric in self.metrics.values():
            metric.update(inputs, targets, extras, predictions, ctx)

    def batch_weight(self, targets: torch.Tensor) -> float:
        """Return the weight this batch contributes to an accumulation window.

        Args:
            targets: Batch targets, which carry the weight (token count, say).

        Returns:
            float: Weight reported by the loss tracker.
        """
        return self.loss_tracker.batch_weight(targets)

    def loss(self) -> float:
        """Return the loss aggregated over the pass so far.

        Returns:
            float: Value computed by the loss tracker.
        """
        return self.loss_tracker.compute()

    def values(self) -> dict[str, float]:
        """Return every configured metric's value, flattened into scalars.

        Returns:
            dict[str, float]: Metric name to value, with nested metrics
                flattened into one entry per scalar.
        """
        values: dict[str, float] = {}

        for name, metric in self.metrics.items():
            values.update(flatten_metric(name, metric.compute()))

        return values

    def record(
        self,
        history: History,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> dict[str, str]:
        """Append one epoch's metric values to `history`.

        Args:
            history: Training history, extended in place.
            train_metrics: Metric values from the epoch's training pass.
            val_metrics: Metric values from the epoch's validation pass, or
                None if no validation loader was used.

        Returns:
            dict[str, str]: Formatted metric values, for the progress bar.
        """
        display: dict[str, str] = {}

        for name, value in train_metrics.items():
            history.setdefault(f"train_{name}", []).append(value)
            display[name] = f"{value:.4g}"

        for name, value in (val_metrics or {}).items():
            history.setdefault(f"val_{name}", []).append(value)
            display[f"val_{name}"] = f"{value:.4g}"

        return display

    def running_postfix(self, history: History) -> dict[str, str]:
        """Return the progress bar postfix for the batch just finished.

        Args:
            history: Training history, read for the previous epochs' values.

        Returns:
            dict[str, str]: Formatted running loss, metrics and history tail.
        """
        postfix = {"train_loss": f"{self.loss():.4g}"}

        for name, value in self.values().items():
            postfix[name] = f"{value:.4g}"

        for key, values in history.items():
            if key not in postfix and values:
                postfix[key] = f"{values[-1]:.4g}"

        return postfix
