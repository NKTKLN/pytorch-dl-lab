"""The series a run records, and the axis their values sit on."""

from collections.abc import Mapping

from dl_roadmap.engine.trainer.context import AxisUnit

History = dict[str, list[float]]
"""Loss and metric series, one value per recorded interval."""


class TrainingHistory(dict[str, list[float]]):
    """Recorded series, and where along the run each value was recorded."""

    def __init__(self, series: History | None = None) -> None:
        """Start from the given series, or from empty loss series.

        Args:
            series: Series to start from, as `dict(history)` returns them.
                None starts with empty "train_loss" and "val_loss".
        """
        if series is None:
            series = {"train_loss": [], "val_loss": []}

        super().__init__(series)

        self.axis: list[int] = []
        self.unit: AxisUnit = "epoch"

    def append(self, position: int, values: Mapping[str, float]) -> None:
        """Record one interval's values, all sitting at `position`.

        Args:
            position: End of the interval just recorded, counted in `unit`s:
                the epoch number under an epoch schedule, and the optimizer
                step or token count under the others.
            values: Series name to the value this interval recorded for it.
                Every series is expected on every interval; one that skips an
                interval no longer lines up with the axis.
        """
        self.axis.append(position)

        for name, value in values.items():
            self.setdefault(name, []).append(value)
