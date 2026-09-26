"""What a run is divided into, and how each division counts its work."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, count, repeat
from operator import length_hint
from typing import Protocol

from dl_roadmap.engine.trainer.batch import Batch
from dl_roadmap.engine.trainer.context import AxisUnit, Unit
from dl_roadmap.engine.trainer.optimization import Optimization


def one_batch(_batch: Batch) -> int:
    """Count a batch as a single unit of work.

    Args:
        _batch: Batch just processed; unused.

    Returns:
        int: Always 1.
    """
    return 1


@dataclass(frozen=True)
class Interval[BatchT: Batch]:
    """One stretch of a run, ending where the model is evaluated.

    Attributes:
        number: Position of the stretch in the run, counted from 1. Under an
            epoch schedule it is the epoch number.
        batches: Batches the stretch consumes.
        total: Work it spans, counted in `unit`; None when unknown.
        unit: What `total` and `units` count.
        units: Work one batch consumes, in `unit`s.
        position: Where the stretch ends on the history axis, in `unit`s.
            None records `number` instead, which is what an epoch schedule
            wants. A step or token schedule plans it in advance, so a stretch
            that ends early records the plan rather than the truth.
    """

    number: int
    batches: Iterable[BatchT]
    total: int | None = None
    unit: Unit = "batch"
    units: Callable[[BatchT], int] = one_batch
    position: int | None = None

    @property
    def axis_unit(self) -> AxisUnit:
        """Return what this interval's position is counted in.

        Returns:
            AxisUnit: The interval's own unit when it plans a position, and
                "epoch" when the run counts plain intervals.
        """
        return "epoch" if self.position is None else self.unit


class TrainingSchedule[BatchT: Batch](Protocol):
    """What a run consists of and how its work is counted."""

    def __len__(self) -> int:
        """Return how many intervals the whole run spans.

        Returns:
            int: Interval count, or 0 when the run has no planned end.
        """

    def intervals(
        self, loader: Iterable[BatchT], start: int = 1
    ) -> Iterator[Interval[BatchT]]:
        """Yield the stretches the run is divided into.

        Args:
            loader: Training batches the run draws from.
            start: Index of the first stretch to run, counted from 1, so that
                a run can resume where a checkpoint left it.

        Returns:
            Iterator[Interval[BatchT]]: The stretches, in order.
        """


class EpochSchedule[BatchT: Batch]:
    """Divide a run into epochs, each one pass over the loader."""

    def __init__(self, epochs: int) -> None:
        """Bind the number of epochs the run spans.

        Args:
            epochs: Passes over the training loader; must be >= 1.

        Raises:
            ValueError: If `epochs` is below 1.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}.")

        self.epochs = epochs

    def __len__(self) -> int:
        """Return how many epochs the run spans.

        Returns:
            int: The configured epoch count.
        """
        return self.epochs

    def intervals(
        self, loader: Iterable[BatchT], start: int = 1
    ) -> Iterator[Interval[BatchT]]:
        """Yield one interval per remaining epoch.

        Args:
            loader: Training batches, walked once per epoch.
            start: First epoch to run, counted from 1.

        Returns:
            Iterator[Interval[BatchT]]: One interval per epoch from `start`.

        Raises:
            ValueError: If `start` is outside 1..`epochs`.
        """
        if not 1 <= start <= self.epochs:
            raise ValueError(f"start must be in 1..{self.epochs}, got {start}.")

        return (
            Interval(
                number=epoch,
                batches=loader,
                total=length_hint(loader) or None,
            )
            for epoch in range(start, self.epochs + 1)
        )


class StepSchedule[BatchT: Batch]:
    """Divide a run into windows of optimizer steps over an endless stream."""

    def __init__(
        self, optimization: Optimization, max_steps: int, eval_every: int
    ) -> None:
        """Bind the engine whose steps are counted, and the budget.

        Args:
            optimization: Engine the trainer steps. Its `steps` counter is the
                only source of truth for progress, since a batch under
                gradient accumulation may take no step at all.
            max_steps: Optimizer steps the whole run spans; must be >= 1.
            eval_every: Steps between evaluations; must be >= 1.

        Raises:
            ValueError: If either budget is below 1.
        """
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}.")

        if eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {eval_every}.")

        self.optimization = optimization
        self.max_steps = max_steps
        self.eval_every = eval_every
        self._counted = 0

    def __len__(self) -> int:
        """Return how many evaluation windows the run spans.

        Returns:
            int: Windows the step budget is divided into.
        """
        return (self.max_steps + self.eval_every - 1) // self.eval_every

    def intervals(
        self, loader: Iterable[BatchT], start: int = 1
    ) -> Iterator[Interval[BatchT]]:
        """Yield one window per evaluation, over a loader walked endlessly.

        Args:
            loader: Training batches, cycled until the budget runs out.
            start: First window to run, counted from 1.

        Returns:
            Iterator[Interval[BatchT]]: Windows of at most `eval_every` steps.
        """
        stream = chain.from_iterable(repeat(loader))

        for number in count(start):
            spent = (number - 1) * self.eval_every
            window = min(self.eval_every, self.max_steps - spent)

            if window <= 0:
                return

            yield Interval(
                number=number,
                batches=self._window(stream, window),
                total=window,
                unit="step",
                units=self.units,
                position=spent + window,
            )

    def _window(self, stream: Iterator[BatchT], window: int) -> Iterator[BatchT]:
        """Yield batches until the optimizer has taken `window` more steps.

        Args:
            stream: Endless batches to draw from.
            window: Optimizer steps this interval spans.

        Returns:
            Iterator[BatchT]: The batches the window consumes.
        """
        self._counted = self.optimization.steps
        target = self.optimization.steps + window

        for batch in stream:
            yield batch

            if self.optimization.steps >= target:
                return

    def units(self, _batch: BatchT) -> int:
        """Return how many optimizer steps the batch just processed took.

        Args:
            _batch: Batch just processed; the count comes from the engine.

        Returns:
            int: Steps taken since the previous call, so 0 for a batch that
                only accumulated gradients. Call it once per batch.
        """
        taken = self.optimization.steps
        advance = taken - self._counted
        self._counted = taken

        return advance


class TokenSchedule[BatchT: Batch]:
    """Divide a run into windows of a token budget over an endless stream."""

    def __init__(self, budget: int, eval_every: int, pad_id: int | None = None) -> None:
        """Bind the token budget and how tokens are counted.

        Args:
            budget: Tokens the whole run spans; must be >= 1.
            eval_every: Tokens between evaluations; must be >= 1.
            pad_id: Token id that padding uses, which is not counted. None
                counts every position of the batch inputs.

        Raises:
            ValueError: If either budget is below 1.
        """
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}.")

        if eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {eval_every}.")

        self.budget = budget
        self.eval_every = eval_every
        self.pad_id = pad_id
        self._counted = 0

    def __len__(self) -> int:
        """Return how many evaluation windows the run spans.

        Returns:
            int: Windows the token budget is divided into.
        """
        return (self.budget + self.eval_every - 1) // self.eval_every

    def intervals(
        self, loader: Iterable[BatchT], start: int = 1
    ) -> Iterator[Interval[BatchT]]:
        """Yield one window per evaluation, over a loader walked endlessly.

        Args:
            loader: Training batches, cycled until the budget runs out.
            start: First window to run, counted from 1.

        Returns:
            Iterator[Interval[BatchT]]: Windows of about `eval_every` tokens;
                the last batch of a window may overshoot it.
        """
        stream = chain.from_iterable(repeat(loader))

        for number in count(start):
            spent = (number - 1) * self.eval_every
            window = min(self.eval_every, self.budget - spent)

            if window <= 0:
                return

            yield Interval(
                number=number,
                batches=self._window(stream, window),
                total=window,
                unit="token",
                units=self.units,
                position=spent + window,
            )

    def _window(self, stream: Iterator[BatchT], window: int) -> Iterator[BatchT]:
        """Yield batches until `window` tokens have been consumed.

        Args:
            stream: Endless batches to draw from.
            window: Tokens this interval spans.

        Returns:
            Iterator[BatchT]: The batches the window consumes.
        """
        spent = 0

        for batch in stream:
            self._counted = self._tokens(batch)
            spent += self._counted

            yield batch

            if spent >= window:
                return

    def units(self, _batch: BatchT) -> int:
        """Return how many tokens the batch just processed carried.

        Args:
            _batch: Batch just processed; it was counted as it was yielded.

        Returns:
            int: Tokens of the batch the window yielded last. Call it once per
                batch.
        """
        return self._counted

    def _tokens(self, batch: BatchT) -> int:
        """Count the tokens a batch's inputs carry.

        Args:
            batch: Batch to count, whose first tensor holds the inputs.

        Returns:
            int: Positions of the inputs, without the padded ones.
        """
        inputs = batch[0]

        if self.pad_id is None:
            return int(inputs.numel())

        return int(inputs.ne(self.pad_id).sum())
