"""When a run writes its state to disk, and where."""

from pathlib import Path
from typing import Protocol

from loguru import logger

from dl_roadmap.engine.trainer.state_store import TrainerStateStore


class Checkpointer(Protocol):
    """Decide whether a finished interval is worth saving."""

    def after_interval(self, number: int, store: TrainerStateStore) -> Path | None:
        """Save the run's state if this interval calls for it.

        Args:
            number: Interval that just finished, counted from 1.
            store: State to write, which owns the model and the history.

        Returns:
            Path | None: Where the state was written, or None if this
                interval was not saved.
        """


class NoCheckpoints:
    """Keep nothing, for a run that needs no intermediate state."""

    def after_interval(self, _number: int, _store: TrainerStateStore) -> Path | None:
        """Save nothing.

        Args:
            _number: Interval that just finished; unused.
            _store: State that would be written; unused.

        Returns:
            Path | None: Always None.
        """
        return None


_NUMBER_WIDTH = 4
"""Digits an epoch or interval number is padded to, so that files sort."""

_COUNT_WIDTH = 8
"""Digits a step or token count is padded to, for the same reason."""


class EveryNIntervals:
    """Write the run's state into a directory every N intervals."""

    def __init__(self, directory: str | Path, every: int = 1) -> None:
        """Bind where checkpoints go and how often they are written.

        Args:
            directory: Directory the files are written into, created on the
                first save.
            every: Intervals between saves; must be >= 1.

        Raises:
            ValueError: If `every` is below 1.
        """
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}.")

        self.directory = Path(directory)
        self.every = every

    def after_interval(self, number: int, store: TrainerStateStore) -> Path | None:
        """Save the state on every `every`-th interval.

        Args:
            number: Interval that just finished, counted from 1.
            store: State to write.

        Returns:
            Path | None: Where the state was written, or None between saves.
                The file is named after the run's axis, so an epoch schedule
                writes "epoch_0004.pt" and a step schedule "step_00002000.pt".
        """
        if number % self.every:
            return None

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self._name(number, store)}.pt"

        store.save(path, number)
        logger.debug(f"Saved checkpoint: {path}")

        return path

    @staticmethod
    def _name(number: int, store: TrainerStateStore) -> str:
        """Return the file name for the interval that just finished.

        Args:
            number: Interval that just finished, counted from 1.
            store: State to write, read for the axis the run records on.

        Returns:
            str: The axis unit and the position reached, or the interval
                number when nothing has been recorded yet.
        """
        history = store.history

        if not history.axis:
            return f"interval_{number:0{_NUMBER_WIDTH}d}"

        width = _NUMBER_WIDTH if history.unit == "epoch" else _COUNT_WIDTH

        return f"{history.unit}_{history.axis[-1]:0{width}d}"
