"""Orchestrate training: run the loop and delegate the work to collaborators."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from operator import length_hint
from typing import cast

import torch
from loguru import logger
from torch import nn

from dl_roadmap.engine.trainer.batch import Batch, BatchParts
from dl_roadmap.engine.trainer.checkpointer import Checkpointer, NoCheckpoints
from dl_roadmap.engine.trainer.context import AxisUnit, Phase, StepContext
from dl_roadmap.engine.trainer.early_stopping import EarlyStopping
from dl_roadmap.engine.trainer.loss_tracker import LossTracker, MeanLossTracker
from dl_roadmap.engine.trainer.online_metrics import Metric, flatten_metric
from dl_roadmap.engine.trainer.optimization import NoOptimization, Optimization
from dl_roadmap.engine.trainer.progress import ProgressReporter, TqdmProgress
from dl_roadmap.engine.trainer.report import IntervalReport
from dl_roadmap.engine.trainer.schedule import Interval, TrainingSchedule
from dl_roadmap.engine.trainer.state_store import TrainerStateStore

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""Compute a scalar loss from predictions and targets, in that order."""

IntervalCallback = Callable[[IntervalReport], None]
"""Called once per finished interval, after the learning-rate schedule."""


class Trainer[BatchT: Batch]:
    """Train and evaluate supervised PyTorch models."""

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        loss_fn: LossFn,
        optimization: Optimization | None = None,
        loss_tracker: LossTracker | None = None,
        metrics: Mapping[str, Metric] | None = None,
        progress: ProgressReporter | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Wire the model to the components that train and measure it.

        Args:
            model: Model to train; moved to `device`.
            loss_fn: Compute a scalar loss from predictions and targets.
            optimization: Engine owning the optimizer step. None installs
                `NoOptimization`, leaving the trainer in evaluation mode:
                `evaluate` and `predict` work, `fit` raises.
            loss_tracker: Aggregate batch losses over a pass; must match
                `loss_fn`'s reduction, e.g. `PerTokenLossTracker` for a
                token-summed loss. None averages them with `MeanLossTracker`.
            metrics: Named metrics accumulated during training and
                validation. Results are stored in the history under
                `train_<name>` and `val_<name>`.
            progress: Sink every pass reports to. None draws a tqdm bar;
                `NullProgress` runs silently.
            device: Device the model and every batch are moved to. None
                requests CUDA when it is available, otherwise CPU.
        """
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimization = optimization or NoOptimization()
        self.loss_tracker = loss_tracker or MeanLossTracker()
        self.metrics = dict(metrics or {})
        self.progress: ProgressReporter = progress or TqdmProgress()

        self.optimization.prepare(self.device)

        self.state_store = TrainerStateStore(
            self.model, self.optimization, map_location=self.device
        )

        logger.debug(
            f"Trainer initialized: model={type(model).__name__}, "
            f"device={self.device}, "
            f"optimization={'on' if self.optimization.can_step else 'off'}"
        )

    def _forward(self, batch: BatchT, _ctx: StepContext) -> torch.Tensor:
        """Compute model predictions for a batch.

        Args:
            batch: Every tensor of the batch, already on `self.device`, with
                its own type — field names included for a `NamedTuple` batch.
            _ctx: Phase, interval and step the batch belongs to; unused here.

        Returns:
            torch.Tensor: Model predictions passed to `self.loss_fn`.
        """
        return self.model(self._split(batch).inputs)  # type: ignore[no-any-return]

    def _split(self, batch: BatchT) -> BatchParts:
        """Name a batch's tensors by the role the trainer reads them in.

        Args:
            batch: Every tensor of the batch, already on `self.device`.

        Returns:
            BatchParts: The first tensor as inputs and the second as
                targets. Override for a batch ordered otherwise.
        """
        inputs, *rest = batch

        return BatchParts(inputs, rest[0] if rest else None)

    def _to_device(self, batch: BatchT) -> BatchT:
        """Move every tensor of a batch to `self.device`, keeping its type.

        Args:
            batch: Batch as the loader yielded it.

        Returns:
            BatchT: The same kind of batch, tensor by tensor on the device.
                A `NamedTuple` is rebuilt through `_make`, so it keeps its
                field names; anything else comes back a plain tuple.
        """
        moved = [tensor.to(self.device) for tensor in batch]
        make = getattr(batch, "_make", None)

        if make is not None:
            return cast(BatchT, make(moved))

        return cast(BatchT, tuple(moved))

    def _record_interval(
        self,
        position: int,
        ctx: StepContext,
        train_loss: float,
        train_metrics: dict[str, float],
        val_loss: float | None,
        val_metrics: dict[str, float] | None,
    ) -> IntervalReport:
        """Append an interval's results to the history and report them.

        Args:
            position: Where the interval just finished ends on the history
                axis: the epoch number, or the step or token count.
            ctx: Where in the run the interval sits, passed on to the
                callbacks watching it.
            train_loss: Average loss over the interval's training pass.
            train_metrics: Metric values from the same pass.
            val_loss: Average loss over the validation pass, or None when
                validation is disabled.
            val_metrics: Metric values from that pass, or None for the same
                reason.

        Returns:
            IntervalReport: Everything the interval recorded, for the
                learning-rate schedule, the callbacks and early stopping.
        """
        values: dict[str, float] = {"train_loss": train_loss}

        if val_loss is not None:
            values["val_loss"] = val_loss

        values.update({f"train_{name}": value for name, value in train_metrics.items()})
        values.update(
            {f"val_{name}": value for name, value in (val_metrics or {}).items()}
        )

        self.state_store.history.append(position, values)

        return IntervalReport(ctx, train_loss, val_loss, values, self.state_store)

    def fit(  # noqa: PLR0913
        self,
        train_loader: Iterable[BatchT],
        val_loader: Iterable[BatchT] | None = None,
        start_interval: int = 1,
        *,
        schedule: TrainingSchedule[BatchT],
        early_stopping: EarlyStopping | None = None,
        checkpointer: Checkpointer | None = None,
        callbacks: Iterable[IntervalCallback] = (),
    ) -> None:
        """Run the training loop over every interval of `schedule`.

        Args:
            train_loader: Batches of (inputs, targets, *extras) used for
                training.
            val_loader: Optional batches of (inputs, targets, *extras) used
                for per-interval validation. Required if `early_stopping` is
                set.
            start_interval: First interval to run, counted from 1. To resume
                a run, pass the interval number `load()` returns plus one;
                `schedule` keeps the total, so fewer intervals are left.
            schedule: What the run is divided into and how its work is
                counted, e.g. `EpochSchedule(10)`.
            early_stopping: Strategy selecting the best interval and deciding
                when to stop. One built with `restore_best_weights` puts that
                interval's weights back into the model when the run ends.
            checkpointer: Decides which finished intervals are written to
                disk. None keeps nothing.
            callbacks: Functions called once per finished interval, after
                the learning-rate schedule, with the interval's report. One
                that raises ends the run, and the best weights are still
                restored on the way out.

        Raises:
            ValueError: If no `optimization` engine is configured, if
                `early_stopping` is set but no `val_loader` is given, if a
                batch carries no targets, or if the learning-rate schedule
                monitors a value the run does not record.
        """
        if not self.optimization.can_step:
            raise ValueError("fit requires an OptimizationEngine.")

        if early_stopping is not None and val_loader is None:
            raise ValueError("early_stopping requires a val_loader.")

        checkpointer = checkpointer or NoCheckpoints()

        logger.debug(
            f"Starting training: intervals={start_interval}..{len(schedule)}, "
            f"val={'yes' if val_loader is not None else 'no'}, "
            f"checkpointer={type(checkpointer).__name__}"
        )

        try:
            history = self.state_store.history

            for interval in schedule.intervals(train_loader, start_interval):
                number = interval.number
                train_loss, train_metrics = self._run_pass(
                    interval, phase="train", intervals=len(schedule)
                )

                val_loss: float | None = None
                val_metrics: dict[str, float] | None = None
                if val_loader is not None:
                    val_loss, val_metrics = self._run_pass(
                        self._pass_interval(number, val_loader),
                        phase="val",
                        intervals=len(schedule),
                        axis_unit=interval.axis_unit,
                    )

                history.unit = interval.axis_unit

                report = self._record_interval(
                    interval.position or number,
                    StepContext(
                        phase="train",
                        interval=number,
                        intervals=len(schedule),
                        unit=interval.axis_unit,
                        step=self.optimization.steps,
                    ),
                    train_loss,
                    train_metrics,
                    val_loss,
                    val_metrics,
                )

                self.optimization.step_interval(report.values)

                for callback in callbacks:
                    callback(report)

                self.progress.refresh(
                    {name: f"{value:.4g}" for name, value in report.values.items()}
                )

                if early_stopping is not None:
                    early_stopping.update(report)

                checkpointer.after_interval(number, self.state_store)

                if early_stopping is not None and early_stopping.should_stop:
                    break

        finally:
            self.progress.close()

            if early_stopping is not None:
                early_stopping.restore(self.model)

        logger.debug("Training complete")

    def evaluate(self, loader: Iterable[BatchT]) -> tuple[float, dict[str, float]]:
        """Run one evaluation pass, without touching the optimizer.

        Args:
            loader: Batches of (inputs, targets, *extras) to evaluate on.

        Returns:
            tuple[float, dict[str, float]]: The loss `loss_tracker` aggregated
                over the pass and each configured metric's value.

        Raises:
            ValueError: If a batch carries no targets.
        """
        try:
            return self._run_pass(self._pass_interval(0, loader), phase="val")
        finally:
            self.progress.close()

    def predict(self, loader: Iterable[BatchT]) -> torch.Tensor:
        """Collect model predictions for every batch in `loader`.

        Args:
            loader: Batches whose first tensor holds the inputs. The batch
                reaches `_forward` whole, targets included when the loader
                yields them, so an override that must not see them checks
                `ctx.phase`. A loader of inputs alone works too.

        Returns:
            torch.Tensor: Predictions for all batches, concatenated on CPU.

        Raises:
            ValueError: If `loader` yields no batches.
        """
        self.model.eval()
        ctx = StepContext(phase="predict", step=self.optimization.steps)
        outputs: list[torch.Tensor] = []

        self.progress.start_pass(ctx, total=length_hint(loader) or None)

        try:
            with torch.no_grad():
                for raw_batch in loader:
                    batch = self._to_device(raw_batch)

                    with self.optimization.autocast():
                        predictions = self._forward(batch, ctx)

                    outputs.append(predictions.detach().float().cpu())
                    self.progress.update()
        finally:
            self.progress.close()

        return torch.cat(outputs)

    def _pass_interval(self, number: int, loader: Iterable[BatchT]) -> Interval[BatchT]:
        """Wrap a loader as one interval walked batch by batch.

        Args:
            number: Interval the pass belongs to; 0 outside `fit`.
            loader: Batches the pass walks once.

        Returns:
            Interval[BatchT]: The loader as an interval counted in batches.
        """
        return Interval(
            number=number,
            batches=loader,
            total=length_hint(loader) or None,
        )

    def _run_pass(
        self,
        interval: Interval[BatchT],
        phase: Phase,
        intervals: int = 0,
        axis_unit: AxisUnit | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Run a single train or evaluation pass over an interval's batches.

        Args:
            interval: Batches to walk, how much work they are and how it is
                counted.
            phase: "train" runs with gradient updates, "val" runs under
                `torch.no_grad()`.
            intervals: Intervals the run spans; 0 outside of `fit`.
            axis_unit: What the run counts its intervals in, for a pass that
                belongs to another interval, such as validation after a step
                window. None takes the interval's own.

        Returns:
            tuple[float, dict[str, float]]: The loss `loss_tracker` aggregated
                over the pass, and each configured metric's value over it.

        Raises:
            ValueError: If a batch carries no targets to compare against.
        """
        is_training = phase == "train"
        logger.debug(f"Running {phase} pass for interval {interval.number}")

        self.model.train(mode=is_training)
        self.loss_tracker.reset()

        for metric in self.metrics.values():
            metric.reset()

        pass_ctx = StepContext(
            phase=phase,
            interval=interval.number,
            intervals=intervals,
            unit=axis_unit or interval.axis_unit,
            step=self.optimization.steps,
        )
        self.progress.start_pass(pass_ctx, total=interval.total, unit=interval.unit)

        optimization = self.optimization
        weigh_by_loss = (
            is_training and optimization.config.grad_normalizer == "loss_weights"
        )

        with torch.enable_grad() if is_training else torch.no_grad():
            for raw_batch in interval.batches:
                batch = self._to_device(raw_batch)
                parts = self._split(batch)

                if parts.targets is None:
                    raise ValueError(
                        f"{phase} batches need targets; got a batch of "
                        f"{len(batch)} tensor(s)."
                    )

                ctx = replace(pass_ctx, step=self.optimization.steps)

                with self.optimization.autocast():
                    predictions = self._forward(batch, ctx)
                    loss = self.loss_fn(predictions, parts.targets)

                if is_training:
                    weight = (
                        self.loss_tracker.batch_weight(parts) if weigh_by_loss else 1.0
                    )
                    optimization.backward(loss, weight)
                    optimization.step_if_ready()

                predictions = predictions.detach()
                self.loss_tracker.update(loss.detach().float(), parts, predictions, ctx)

                for metric in self.metrics.values():
                    metric.update(parts, predictions, ctx)

                self.progress.update(
                    self._running_postfix(ctx), advance=interval.units(batch)
                )

        if is_training:
            optimization.flush()

        avg_loss = self.loss_tracker.compute()
        values: dict[str, float] = {}

        for name, metric in self.metrics.items():
            values.update(flatten_metric(name, metric.compute()))

        metrics_repr = "".join(
            f", {name}={value:.4f}" for name, value in values.items()
        )
        logger.debug(f"Finished {phase} pass: loss={avg_loss:.4f}{metrics_repr}")

        return avg_loss, values

    def _running_postfix(self, ctx: StepContext) -> dict[str, str]:
        """Show running loss and recorded results without computing metrics.

        Args:
            ctx: Current pass, used to label its loss as train or validation.

        Returns:
            dict[str, str]: Running loss and the latest value of each series
                recorded for completed intervals.
        """
        postfix = {f"{ctx.phase}_loss": f"{self.loss_tracker.compute():.4g}"}

        for key, values in self.state_store.history.items():
            if key not in postfix and values:
                postfix[key] = f"{values[-1]:.4g}"

        return postfix
