"""The results and state a finished interval makes available."""

from dataclasses import dataclass

from dl_roadmap.engine.trainer.context import StepContext
from dl_roadmap.engine.trainer.state_store import TrainerStateStore


@dataclass(frozen=True)
class IntervalReport:
    """What a finished interval produced.

    Attributes:
        ctx: Where in the run the interval sits, and in what unit.
        train_loss: Average loss over the interval's training pass.
        val_loss: Average loss over its validation pass, or None when
            validation is disabled.
        values: Everything recorded for the interval, under the keys the
            history stores it by: "train_loss", "val_loss" and a
            "train_<name>"/"val_<name>" pair per metric.
        store: The run's state, for reading the history or saving on demand.
    """

    ctx: StepContext
    train_loss: float
    val_loss: float | None
    values: dict[str, float]
    store: TrainerStateStore
