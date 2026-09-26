"""History, state store, checkpointing, progress and step context."""

from pathlib import Path

import pytest
import torch
from helpers import make_trainer, regression_loader
from torch import nn

from dl_roadmap.engine.trainer import (
    EpochSchedule,
    StepContext,
    TrainerStateStore,
    TrainingHistory,
)
from dl_roadmap.engine.trainer.checkpointer import EveryNIntervals, NoCheckpoints
from dl_roadmap.engine.trainer.progress import NullProgress, TqdmProgress


def test_step_context_knows_the_training_phase() -> None:
    assert StepContext(phase="train").is_training
    assert not StepContext(phase="val").is_training


class TestHistory:
    def test_starts_with_empty_loss_series(self) -> None:
        history = TrainingHistory()

        assert (dict(history), history.axis, history.unit) == (
            {"train_loss": [], "val_loss": []},
            [],
            "epoch",
        )

    def test_appends_values_and_new_series_at_a_position(self) -> None:
        history = TrainingHistory()
        history.append(3, {"train_loss": 1.0, "val_acc": 0.5})

        assert history.axis == [3]
        assert (history["train_loss"], history["val_acc"]) == ([1.0], [0.5])


class TestStateStore:
    def test_round_trips_model_optimizer_and_history(self, tmp_path: Path) -> None:
        trainer, model, engine = make_trainer(momentum=0.9)
        trainer.fit(regression_loader(), schedule=EpochSchedule(2))
        path = trainer.state_store.save(tmp_path / "nested" / "state.pt", 2)

        restored, restored_model, restored_engine = make_trainer(momentum=0.9, seed=1)

        assert restored.state_store.load(path) == 2
        torch.testing.assert_close(restored_model.weight, model.weight)
        assert restored_engine.steps == engine.steps
        assert dict(restored.state_store.history) == dict(trainer.state_store.history)
        assert restored.state_store.history.axis == [1, 2]

    def test_rebuilds_the_axis_of_older_files(self, tmp_path: Path) -> None:
        path = tmp_path / "old.pt"
        model = nn.Linear(1, 1)
        torch.save(
            {
                "epoch": 2,
                "model_state_dict": model.state_dict(),
                "history": {"train_loss": [1.0, 0.5], "val_loss": []},
            },
            path,
        )
        store = TrainerStateStore(nn.Linear(1, 1))
        store.load(path)

        assert (store.history.axis, store.history.unit) == ([1, 2], "epoch")

    def test_saves_no_optimizer_state_without_an_engine(self, tmp_path: Path) -> None:
        path = TrainerStateStore(nn.Linear(1, 1)).save(tmp_path / "s.pt")

        assert "optimization_state_dict" not in torch.load(path)

    def test_refuses_missing_and_foreign_files(self, tmp_path: Path) -> None:
        store = TrainerStateStore(nn.Linear(1, 1))

        with pytest.raises(FileNotFoundError):
            store.load(tmp_path / "absent.pt")

        torch.save({"weights": 1}, tmp_path / "foreign.pt")

        with pytest.raises(KeyError, match="Not a trainer state file"):
            store.load(tmp_path / "foreign.pt")


class TestCheckpointer:
    def test_saves_every_nth_interval_named_by_the_axis(self, tmp_path: Path) -> None:
        trainer, *_ = make_trainer()
        trainer.fit(
            regression_loader(),
            schedule=EpochSchedule(5),
            checkpointer=EveryNIntervals(tmp_path, every=2),
        )

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "epoch_0002.pt",
            "epoch_0004.pt",
        ]

    def test_names_by_interval_before_anything_is_recorded(
        self, tmp_path: Path
    ) -> None:
        path = EveryNIntervals(tmp_path).after_interval(
            3, TrainerStateStore(nn.Linear(1, 1))
        )

        assert path == tmp_path / "interval_0003.pt"

    def test_pads_step_counts_wider(self, tmp_path: Path) -> None:
        store = TrainerStateStore(nn.Linear(1, 1))
        store.history.unit = "step"
        store.history.append(2000, {"train_loss": 1.0})

        path = EveryNIntervals(tmp_path).after_interval(1, store)

        assert path == tmp_path / "step_00002000.pt"

    def test_rejects_every_below_one(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="every must be >= 1"):
            EveryNIntervals(tmp_path, every=0)

    def test_no_checkpoints_writes_nothing(self) -> None:
        assert (
            NoCheckpoints().after_interval(1, TrainerStateStore(nn.Linear(1, 1)))
            is None
        )


class TestProgress:
    def test_tqdm_reuses_one_bar_across_passes(self) -> None:
        progress = TqdmProgress()
        progress.update({"x": "1"})
        progress.start_pass(
            StepContext(phase="train", interval=1, intervals=10), total=4
        )
        bar = progress._bar
        progress.update({"loss": "1"}, advance=2)
        progress.start_pass(StepContext(phase="val", interval=1, intervals=10), total=3)

        assert progress._bar is bar
        assert bar is not None
        assert (bar.n, bar.total, bar.desc) == (0, 3, "epoch  1/10 val")

        progress.close()

        assert progress._bar is None

    def test_label_drops_the_counter_outside_fit(self) -> None:
        assert TqdmProgress._description(StepContext(phase="predict")) == "predict"

    def test_null_progress_accepts_everything(self) -> None:
        progress = NullProgress()
        progress.start_pass(StepContext(phase="train"), total=None, unit="token")
        progress.update({"a": "b"}, advance=3)
        progress.refresh({"a": "b"})
        progress.close()
