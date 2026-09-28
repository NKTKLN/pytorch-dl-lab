"""The trainer end to end: fit, evaluate, predict, resuming and hooks."""

from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from helpers import RecordingProgress, make_trainer, regression_loader
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset

from dl_roadmap.engine.trainer import (
    BatchParts,
    EpochSchedule,
    IntervalReport,
    Metric,
    OptimizationConfig,
    OptimizationEngine,
    PerTokenLossTracker,
    Phase,
    StepContext,
    TorchLRSchedule,
    Trainer,
    ValLossEarlyStopping,
)
from dl_roadmap.engine.trainer.checkpointer import EveryNIntervals
from dl_roadmap.engine.trainer.progress import NullProgress
from dl_roadmap.engine.trainer.schedule import StepSchedule
from dl_roadmap.engine.trainers import TeacherForcingTrainer


class MeanAbsError(Metric):
    def reset(self) -> None:
        self.total, self.count = 0.0, 0

    def update(
        self, parts: BatchParts, predictions: torch.Tensor, ctx: StepContext
    ) -> None:
        del ctx
        errors = (predictions - parts.require_targets()).abs()
        self.total += float(errors.sum())
        self.count += errors.numel()

    def compute(self) -> float:
        return self.total / self.count


class RecordingMetric(Metric):
    """Remembers the context of every batch it was shown."""

    def __init__(self) -> None:
        self.contexts: list[StepContext] = []

    def reset(self) -> None:
        pass

    def update(
        self, parts: BatchParts, predictions: torch.Tensor, ctx: StepContext
    ) -> None:
        del parts, predictions
        self.contexts.append(ctx)

    def compute(self) -> float:
        return 0.0


class TestFitArguments:
    def test_needs_an_optimization_engine(self) -> None:
        trainer: Trainer[tuple[torch.Tensor, ...]] = Trainer(
            nn.Linear(3, 1), nn.MSELoss(), progress=NullProgress(), device="cpu"
        )

        with pytest.raises(ValueError, match="requires an OptimizationEngine"):
            trainer.fit(regression_loader(), schedule=EpochSchedule(1))

    def test_early_stopping_needs_validation(self) -> None:
        trainer, *_ = make_trainer()

        with pytest.raises(ValueError, match="requires a val_loader"):
            trainer.fit(
                regression_loader(),
                schedule=EpochSchedule(1),
                early_stopping=ValLossEarlyStopping(1),
            )

    def test_batches_without_targets_are_refused(self) -> None:
        trainer, *_ = make_trainer()
        loader = DataLoader(TensorDataset(torch.randn(4, 3)), batch_size=2)

        with pytest.raises(ValueError, match="batches need targets"):
            trainer.fit(loader, schedule=EpochSchedule(1))


class TestFit:
    def test_learns_a_linear_target(self) -> None:
        trainer, *_ = make_trainer()
        trainer.fit(
            regression_loader(), regression_loader(seed=1), schedule=EpochSchedule(20)
        )
        history = trainer.state_store.history

        assert history["val_loss"][-1] < 1e-3
        assert history.axis == list(range(1, 21))

    def test_records_metrics_under_their_phase(self) -> None:
        trainer, *_ = make_trainer()
        trainer.metrics = {"mae": MeanAbsError()}
        trainer.fit(
            regression_loader(), regression_loader(seed=1), schedule=EpochSchedule(2)
        )

        assert set(trainer.state_store.history) == {
            "train_loss",
            "val_loss",
            "train_mae",
            "val_mae",
        }

    def test_callbacks_see_every_interval_after_the_lr_schedule(self) -> None:
        trainer, _, engine = make_trainer()
        engine.scheduler = TorchLRSchedule(StepLR(engine.optimizer, 1, gamma=0.5))
        seen: list[tuple[int, float]] = []

        def callback(report: IntervalReport) -> None:
            seen.append((report.ctx.interval, engine.optimizer.param_groups[0]["lr"]))

        trainer.fit(
            regression_loader(), schedule=EpochSchedule(3), callbacks=[callback]
        )

        assert seen == [(1, 0.05), (2, 0.025), (3, 0.0125)]

    def test_plateau_can_watch_a_validation_metric(self) -> None:
        trainer, _, engine = make_trainer()
        trainer.metrics = {"mae": MeanAbsError()}
        plateau = ReduceLROnPlateau(engine.optimizer, mode="max", patience=0)
        engine.scheduler = TorchLRSchedule(plateau, monitor="val_mae")
        trainer.fit(
            regression_loader(), regression_loader(seed=1), schedule=EpochSchedule(3)
        )

        assert engine.optimizer.param_groups[0]["lr"] < 0.1

    def test_a_failing_callback_ends_the_run_with_best_weights(self) -> None:
        progress = RecordingProgress()
        trainer, model, _ = make_trainer(lr=0.05, progress=progress)
        snapshots: dict[int, torch.Tensor] = {}
        strategy = ValLossEarlyStopping(10, restore_best_weights=True)

        def callback(report: IntervalReport) -> None:
            snapshots[report.ctx.interval] = model.weight.detach().clone()
            if report.ctx.interval == 3:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            trainer.fit(
                regression_loader(),
                regression_loader(seed=1, sign=-1.0),
                schedule=EpochSchedule(5),
                early_stopping=strategy,
                callbacks=[callback],
            )

        assert progress.closed >= 1
        assert strategy.best_interval is not None
        torch.testing.assert_close(model.weight, snapshots[strategy.best_interval])

    def test_early_stopping_ends_the_run_on_the_best_weights(self) -> None:
        trainer, model, _ = make_trainer(lr=0.05)
        strategy = ValLossEarlyStopping(2, restore_best_weights=True)
        snapshots: dict[int, torch.Tensor] = {}
        trainer.fit(
            regression_loader(),
            regression_loader(seed=1, sign=-1.0),
            schedule=EpochSchedule(20),
            early_stopping=strategy,
            callbacks=[
                lambda r: snapshots.__setitem__(
                    r.ctx.interval, model.weight.detach().clone()
                )
            ],
        )

        assert strategy.should_stop
        assert len(trainer.state_store.history.axis) == 3
        torch.testing.assert_close(model.weight, snapshots[1])

    def test_the_stopping_interval_is_still_checkpointed(self, tmp_path: Path) -> None:
        trainer, *_ = make_trainer(lr=0.05)
        trainer.fit(
            regression_loader(),
            regression_loader(seed=1, sign=-1.0),
            schedule=EpochSchedule(20),
            early_stopping=ValLossEarlyStopping(2),
            checkpointer=EveryNIntervals(tmp_path),
        )

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "epoch_0001.pt",
            "epoch_0002.pt",
            "epoch_0003.pt",
        ]


class TestPasses:
    def test_loss_weights_normalize_the_window_by_tokens(self) -> None:
        config = OptimizationConfig(
            amp="off", accumulation_steps=2, grad_normalizer="loss_weights"
        )
        trainer, model, _ = make_trainer(config=config)
        trainer.loss_tracker = PerTokenLossTracker(pad_id=-1)
        trainer.loss_fn = nn.MSELoss(reduction="sum")
        loader = regression_loader(n=5, batch_size=3)
        x, y = loader.dataset[:]

        _, reference, reference_engine = make_trainer()
        reference_engine.backward(
            nn.functional.mse_loss(reference(x), y, reduction="sum") / 5
        )
        reference_engine.step_if_ready()

        trainer.fit(loader, schedule=EpochSchedule(1))

        torch.testing.assert_close(model.weight, reference.weight)
        torch.testing.assert_close(model.bias, reference.bias)

    def test_only_training_builds_a_graph(self) -> None:
        trainer, model, _ = make_trainer()
        modes: list[bool] = []
        model.register_forward_hook(lambda *_: modes.append(torch.is_grad_enabled()))
        loader = regression_loader(n=4, batch_size=4)

        trainer.fit(loader, loader, schedule=EpochSchedule(1))
        trainer.evaluate(loader)

        assert modes == [True, False, False]

    def test_every_pass_knows_where_it_runs(self) -> None:
        progress = RecordingProgress()
        trainer, *_ = make_trainer(progress=progress)
        metric = RecordingMetric()
        trainer.metrics = {"seen": metric}
        reports: list[StepContext] = []
        loader = regression_loader(n=8, batch_size=4)

        trainer.fit(
            loader,
            loader,
            schedule=EpochSchedule(2),
            callbacks=[lambda report: reports.append(report.ctx)],
        )
        trainer.evaluate(loader)
        trainer.predict(loader)

        def pass_ctx(phase: Phase, interval: int, step: int) -> StepContext:
            return StepContext(phase, interval, 2 if interval else 0, "epoch", step)

        assert metric.contexts == [
            pass_ctx("train", 1, 0),
            pass_ctx("train", 1, 1),
            pass_ctx("val", 1, 2),
            pass_ctx("val", 1, 2),
            pass_ctx("train", 2, 2),
            pass_ctx("train", 2, 3),
            pass_ctx("val", 2, 4),
            pass_ctx("val", 2, 4),
            pass_ctx("val", 0, 4),
            pass_ctx("val", 0, 4),
        ]
        assert reports == [pass_ctx("train", 1, 2), pass_ctx("train", 2, 4)]
        assert progress.passes == [
            (pass_ctx("train", 1, 0), 2, "batch"),
            (pass_ctx("val", 1, 2), 2, "batch"),
            (pass_ctx("train", 2, 2), 2, "batch"),
            (pass_ctx("val", 2, 4), 2, "batch"),
            (pass_ctx("val", 0, 4), 2, "batch"),
            (StepContext("predict", step=4), 2, "batch"),
        ]
        assert progress.advanced == 12

    def test_running_postfix_adds_the_last_recorded_values(self) -> None:
        progress = RecordingProgress()
        trainer, *_ = make_trainer(progress=progress)
        loader = regression_loader(n=4, batch_size=4)

        trainer.fit(loader, loader, schedule=EpochSchedule(2))
        history = trainer.state_store.history

        assert progress.postfixes[-1] == {
            "val_loss": f"{history['val_loss'][1]:.4g}",
            "train_loss": f"{history['train_loss'][0]:.4g}",
        }


class TestSchedules:
    def test_step_schedule_spends_exactly_its_budget(self) -> None:
        trainer, _, engine = make_trainer(
            config=OptimizationConfig(amp="off", accumulation_steps=2)
        )
        trainer.fit(
            regression_loader(batch_size=8),
            schedule=StepSchedule(engine, max_steps=6, eval_every=3),
        )
        history = trainer.state_store.history

        assert engine.steps == 6
        assert (history.axis, history.unit) == ([3, 6], "step")

    def test_validation_passes_are_labelled_in_the_run_unit(self) -> None:
        progress = RecordingProgress()
        trainer, _, engine = make_trainer(progress=progress)
        trainer.fit(
            regression_loader(),
            regression_loader(seed=1),
            schedule=StepSchedule(engine, max_steps=4, eval_every=2),
        )

        assert {ctx.unit for ctx, *_ in progress.passes} == {"step"}

    def test_resuming_matches_an_uninterrupted_run(self, tmp_path: Path) -> None:
        straight, straight_model, _ = make_trainer(momentum=0.9)
        straight.fit(regression_loader(), schedule=EpochSchedule(4))

        first, *_ = make_trainer(momentum=0.9)
        first.fit(regression_loader(), schedule=EpochSchedule(2))
        path = first.state_store.save(tmp_path / "state.pt", 2)

        resumed, resumed_model, _ = make_trainer(momentum=0.9, seed=1)
        start = resumed.state_store.load(path) + 1
        resumed.fit(
            regression_loader(), start_interval=start, schedule=EpochSchedule(4)
        )

        torch.testing.assert_close(resumed_model.weight, straight_model.weight)
        assert resumed.state_store.history.axis == [1, 2, 3, 4]
        assert dict(resumed.state_store.history) == pytest.approx(
            dict(straight.state_store.history)
        )


class TestEvaluateAndPredict:
    def test_evaluate_leaves_the_weights_alone(self) -> None:
        trainer, model, _ = make_trainer()
        trainer.metrics = {"mae": MeanAbsError()}
        before = model.weight.detach().clone()

        loss, values = trainer.evaluate(regression_loader())

        assert loss > 0
        assert set(values) == {"mae"}
        assert not model.training
        torch.testing.assert_close(model.weight, before)

    def test_evaluate_with_a_per_token_tracker(self) -> None:
        trainer, *_ = make_trainer()
        trainer.loss_tracker = PerTokenLossTracker(pad_id=-1)
        trainer.loss_fn = nn.MSELoss(reduction="sum")
        loader = regression_loader(n=8, batch_size=4)
        x, y = loader.dataset[:]

        loss, _ = trainer.evaluate(loader)

        with torch.no_grad():
            expected = float(
                nn.functional.mse_loss(trainer.model(x), y, reduction="sum")
            )

        expected /= 8
        assert loss == pytest.approx(expected)

    def test_predict_accepts_inputs_alone(self) -> None:
        trainer, *_ = make_trainer()
        loader = DataLoader(TensorDataset(torch.randn(10, 3)), batch_size=4)

        assert trainer.predict(loader).shape == (10, 1)

    def test_predict_on_nothing_raises(self) -> None:
        trainer, *_ = make_trainer()

        with pytest.raises(ValueError, match="non-empty"):
            trainer.predict([])


class Pair(NamedTuple):
    inputs: torch.Tensor
    targets: torch.Tensor


class TestBatches:
    def test_named_tuples_keep_their_type_on_the_device(self) -> None:
        seen: list[type] = []

        class Recording(Trainer[Pair]):
            def _forward(self, batch: Pair, ctx: StepContext) -> torch.Tensor:
                seen.append(type(batch))
                return super()._forward(batch, ctx)

        model = nn.Linear(3, 1)
        trainer = Recording(
            model,
            nn.MSELoss(),
            OptimizationEngine(torch.optim.SGD(model.parameters(), lr=0.1)),
            progress=NullProgress(),
            device="cpu",
        )
        batch = Pair(torch.randn(2, 3), torch.randn(2, 1))
        trainer.fit([batch], schedule=EpochSchedule(1))

        assert seen == [Pair]

    def test_teacher_forcing_feeds_the_decoder_input(self) -> None:
        calls: list[int] = []

        class Seq2Seq(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(1, 1)

            def forward(
                self, source: torch.Tensor, decoder_input: torch.Tensor | None = None
            ) -> torch.Tensor:
                calls.append(0 if decoder_input is None else 1)
                output: torch.Tensor = self.proj(source)
                return output

        model = Seq2Seq()
        trainer: TeacherForcingTrainer[tuple[torch.Tensor, ...]] = (
            TeacherForcingTrainer(
                model,
                nn.MSELoss(),
                OptimizationEngine(torch.optim.SGD(model.parameters(), lr=0.1)),
                progress=NullProgress(),
                device="cpu",
            )
        )
        batch = (torch.randn(2, 1), torch.randn(2, 1), torch.randn(2, 1))
        trainer.fit([batch], [batch], schedule=EpochSchedule(1))
        trainer.predict([batch[:1]])

        assert calls == [1, 1, 0]
