"""Where a training run reports the progress of a pass."""

from collections.abc import Mapping
from typing import Protocol

from tqdm import tqdm

from dl_roadmap.engine.trainer.context import StepContext, Unit

_UNIT_LABELS: dict[Unit, str] = {"batch": "it", "token": "tok", "step": "step"}
"""How tqdm spells each unit."""

_BAR_FORMAT = (
    "{desc} {percentage:3.0f}%[{bar:20}] "
    "{n_fmt}/{total_fmt} :: eta={remaining}{postfix}"
)
"""Layout of the tqdm bar: label, percentage, bar, count, ETA, values."""


class ProgressReporter(Protocol):
    """Sink a pass reports its progress to."""

    def start_pass(
        self, ctx: StepContext, *, total: int | None, unit: Unit = "batch"
    ) -> None:
        """Announce a pass about to run.

        Args:
            ctx: Where in the run the pass sits, for the sink to label it
                with.
            total: Work the pass spans, counted in `unit`; None when the pass
                has no known length.
            unit: What `total` and `update`'s `advance` count.
        """

    def update(
        self,
        values: Mapping[str, str] | None = None,
        *,
        advance: int = 1,
    ) -> None:
        """Report finished work and the values that go with it.

        Args:
            values: Values to show, already formatted by the caller; None
                keeps the ones reported before.
            advance: Work finished since the previous call, in the same unit
                as `start_pass`'s `total`.
        """

    def refresh(self, values: Mapping[str, str] | None = None) -> None:
        """Report values without any work finishing.

        Args:
            values: Values to show, already formatted by the caller; None
                keeps the ones reported before.
        """

    def close(self) -> None:
        """Release whatever the reporter holds."""


class NullProgress:
    """Discard every report, for a run that shows no progress."""

    def start_pass(
        self, _ctx: StepContext, *, total: int | None, unit: Unit = "batch"
    ) -> None:
        """Ignore the pass about to run.

        Args:
            _ctx: Where in the run the pass sits; unused.
            total: Work the pass spans; unused.
            unit: What `total` counts; unused.
        """
        del total, unit

    def update(
        self,
        _values: Mapping[str, str] | None = None,
        *,
        advance: int = 1,
    ) -> None:
        """Ignore the finished work.

        Args:
            _values: Values to show; unused.
            advance: Work finished since the previous call; unused.
        """
        del advance

    def refresh(self, _values: Mapping[str, str] | None = None) -> None:
        """Ignore the values.

        Args:
            _values: Values to show; unused.
        """

    def close(self) -> None:
        """Hold nothing, so release nothing."""


class TqdmProgress:
    """Draw a run's progress as a tqdm bar, one pass at a time."""

    def __init__(self) -> None:
        """Start without a bar; the first pass creates it."""
        self._bar: tqdm | None = None

    def start_pass(
        self, ctx: StepContext, *, total: int | None, unit: Unit = "batch"
    ) -> None:
        """Create the bar, or point the one already drawn at this pass.

        Args:
            ctx: Where in the run the pass sits, which the label is built
                from.
            total: Work the bar spans; None leaves it a counter with no bar.
            unit: What the bar counts; token counts are scaled to k/M.
        """
        description = self._description(ctx)

        if self._bar is None:
            self._bar = tqdm(
                total=total,
                desc=description,
                ascii=" >=",
                bar_format=_BAR_FORMAT,
                leave=True,
                unit=_UNIT_LABELS[unit],
                unit_scale=unit == "token",
            )
            return

        self._bar.unit = _UNIT_LABELS[unit]
        self._bar.unit_scale = unit == "token"
        self._bar.reset(total=total)
        self._bar.set_description_str(description)
        self._bar.set_postfix({})

    def update(
        self,
        values: Mapping[str, str] | None = None,
        *,
        advance: int = 1,
    ) -> None:
        """Advance the bar, showing the values alongside it.

        Args:
            values: Values for the bar's postfix; None leaves it as it is.
            advance: Work added to the bar's count.
        """
        if self._bar is None:
            return

        if values:
            self._bar.set_postfix(dict(values))

        self._bar.update(advance)

    def refresh(self, values: Mapping[str, str] | None = None) -> None:
        """Redraw the bar with the values, leaving its count alone.

        Args:
            values: Values for the bar's postfix; None redraws what is there.
        """
        if self._bar is None:
            return

        if values:
            self._bar.set_postfix(dict(values))

        self._bar.refresh()

    def close(self) -> None:
        """Close the bar and leave it on screen."""
        if self._bar is None:
            return

        self._bar.close()
        self._bar = None

    @staticmethod
    def _description(ctx: StepContext) -> str:
        """Return the bar's label for a pass.

        Args:
            ctx: Where in the run the pass sits; interval 0 drops the
                counter, as `evaluate` and `predict` leave it unset.

        Returns:
            str: The phase alone, or the interval counter and the phase.
        """
        if not ctx.interval:
            return ctx.phase

        width = len(str(ctx.intervals))

        return f"{ctx.unit} {ctx.interval:>{width}}/{ctx.intervals} {ctx.phase}"
