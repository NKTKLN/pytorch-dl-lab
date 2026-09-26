"""Builders the trainer tests share."""

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_roadmap.engine.trainer import (
    IntervalReport,
    LRSchedule,
    OptimizationConfig,
    OptimizationEngine,
    PairBatch,
    StepContext,
    Trainer,
    TrainerStateStore,
)
from dl_roadmap.engine.trainer.context import Unit
from dl_roadmap.engine.trainer.progress import NullProgress, ProgressReporter

WEIGHTS = torch.tensor([[1.0], [2.0], [3.0]])


def regression_loader(
    n: int = 64, batch_size: int = 16, seed: int = 0, sign: float = 1.0
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Batches of a noiseless linear target, in a fixed order."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=generator)

    return DataLoader(TensorDataset(x, sign * (x @ WEIGHTS)), batch_size=batch_size)


def make_trainer(
    lr: float = 0.1,
    momentum: float = 0.0,
    config: OptimizationConfig | None = None,
    scheduler: LRSchedule | torch.optim.lr_scheduler.LRScheduler | None = None,
    progress: ProgressReporter | None = None,
    seed: int = 0,
) -> tuple[Trainer[PairBatch], nn.Linear, OptimizationEngine]:
    """A CPU trainer of a 3 -> 1 linear model under SGD."""
    torch.manual_seed(seed)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    engine = OptimizationEngine(
        optimizer, scheduler, config or OptimizationConfig(amp="off")
    )
    trainer: Trainer[PairBatch] = Trainer(
        model,
        nn.MSELoss(),
        engine,
        progress=progress or NullProgress(),
        device="cpu",
    )

    return trainer, model, engine


def make_report(
    interval: int,
    train_loss: float = 1.0,
    val_loss: float | None = None,
    values: Mapping[str, float] | None = None,
    model: nn.Module | None = None,
) -> IntervalReport:
    """An interval report as `fit` builds it."""
    recorded = {"train_loss": train_loss, **(values or {})}

    if val_loss is not None:
        recorded["val_loss"] = val_loss

    return IntervalReport(
        StepContext(phase="train", interval=interval),
        train_loss,
        val_loss,
        recorded,
        TrainerStateStore(model or nn.Linear(1, 1)),
    )


class RecordingProgress:
    """Progress sink that remembers what it was told."""

    def __init__(self) -> None:
        self.passes: list[tuple[StepContext, int | None, Unit]] = []
        self.advanced = 0
        self.closed = 0

    def start_pass(
        self, ctx: StepContext, *, total: int | None, unit: Unit = "batch"
    ) -> None:
        self.passes.append((ctx, total, unit))

    def update(
        self, values: Mapping[str, str] | None = None, *, advance: int = 1
    ) -> None:
        del values
        self.advanced += advance

    def refresh(self, values: Mapping[str, str] | None = None) -> None:
        del values

    def close(self) -> None:
        self.closed += 1
