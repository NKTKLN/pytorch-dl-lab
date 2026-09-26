"""Early stopping strategies and the best weights they keep."""

import math

import pytest
import torch
from helpers import make_report
from torch import nn

from dl_roadmap.engine.trainer import (
    CombinedEarlyStopping,
    EarlyStopping,
    GapThresholdEarlyStopping,
    GeneralizationGapEarlyStopping,
    MetricEarlyStopping,
    ValLossEarlyStopping,
)


def feed(strategy: EarlyStopping, val_losses: list[float | None]) -> list[bool]:
    """Update with one interval per loss; return each interval's `is_best`."""
    flags = []

    for number, val_loss in enumerate(val_losses, start=1):
        strategy.update(make_report(number, val_loss=val_loss))
        flags.append(strategy.is_best)

    return flags


class TestValLoss:
    def test_tracks_the_best_interval_and_stops_after_patience(self) -> None:
        strategy = ValLossEarlyStopping(patience=2)

        assert feed(strategy, [1.0, 0.8, 0.9]) == [True, True, False]
        assert (strategy.best_interval, strategy.should_stop) == (2, False)

        strategy.update(make_report(4, val_loss=0.95))

        assert strategy.should_stop

    def test_improvement_must_strictly_clear_min_delta(self) -> None:
        strategy = ValLossEarlyStopping(patience=5, min_delta=0.1)

        assert feed(strategy, [1.0, 0.9, 0.91, 0.85]) == [True, False, False, True]

    def test_missing_validation_changes_nothing(self) -> None:
        strategy = ValLossEarlyStopping(patience=1)
        feed(strategy, [1.0])

        assert feed(strategy, [None]) == [False]
        assert (strategy.best_interval, strategy.should_stop) == (1, False)

    def test_nan_loss_is_never_the_best(self) -> None:
        strategy = ValLossEarlyStopping(patience=3)

        flags = feed(strategy, [math.nan, 1.0, 0.5])

        assert flags == [False, True, True]
        assert strategy.best_interval == 3

    @pytest.mark.parametrize("patience", [0, -1])
    def test_rejects_patience_below_one(self, patience: int) -> None:
        with pytest.raises(ValueError, match="patience must be >= 1"):
            ValLossEarlyStopping(patience)


class TestOtherScores:
    def test_metric_in_max_mode(self) -> None:
        strategy = MetricEarlyStopping("val_acc", patience=1)

        for number, acc in enumerate([0.5, 0.7, 0.6], start=1):
            strategy.update(make_report(number, values={"val_acc": acc}))

        assert (strategy.best_interval, strategy.should_stop) == (2, True)

    def test_metric_absent_from_the_report_is_skipped(self) -> None:
        strategy = MetricEarlyStopping("val_acc", patience=1)
        strategy.update(make_report(1))

        assert (strategy.best_interval, strategy.is_best) == (None, False)

    def test_generalization_gap_prefers_a_smaller_gap(self) -> None:
        strategy = GeneralizationGapEarlyStopping(patience=1)
        strategy.update(make_report(1, train_loss=1.0, val_loss=1.5))
        strategy.update(make_report(2, train_loss=1.0, val_loss=1.2))
        strategy.update(make_report(3, train_loss=0.5, val_loss=1.2))

        assert (strategy.best_interval, strategy.should_stop) == (2, True)

    def test_gap_threshold_counts_consecutive_intervals_over_it(self) -> None:
        strategy = GapThresholdEarlyStopping(patience=2, threshold=0.2)
        gaps = [0.3, 0.1, 0.3, 0.3]
        stops = []

        for number, gap in enumerate(gaps, start=1):
            strategy.update(make_report(number, train_loss=1.0, val_loss=1.0 + gap))
            stops.append(strategy.should_stop)

        assert stops == [False, False, False, True]
        assert strategy.best_interval is None

    def test_gap_threshold_rejects_patience_below_one(self) -> None:
        with pytest.raises(ValueError, match="patience must be >= 1"):
            GapThresholdEarlyStopping(0, threshold=0.1)


class TestCombined:
    def test_any_stops_with_the_first_strategy(self) -> None:
        strategy = CombinedEarlyStopping(
            [ValLossEarlyStopping(1), ValLossEarlyStopping(3)], combine="any"
        )
        feed(strategy, [1.0, 1.1])

        assert strategy.should_stop

    def test_all_waits_for_every_strategy(self) -> None:
        strategy = CombinedEarlyStopping(
            [ValLossEarlyStopping(1), ValLossEarlyStopping(3)], combine="all"
        )
        feed(strategy, [1.0, 1.1])

        assert not strategy.should_stop

        feed(strategy, [1.2, 1.3])

        assert strategy.should_stop

    def test_best_follows_any_improving_strategy(self) -> None:
        strategy = CombinedEarlyStopping(
            [ValLossEarlyStopping(5), MetricEarlyStopping("val_acc", patience=5)]
        )
        strategy.update(make_report(1, val_loss=1.0, values={"val_acc": 0.5}))
        strategy.update(make_report(2, val_loss=1.1, values={"val_acc": 0.6}))

        assert (strategy.is_best, strategy.best_interval) == (True, 2)

    def test_rejects_no_strategies(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            CombinedEarlyStopping([])

    def test_rejects_restore_on_a_wrapped_strategy(self) -> None:
        with pytest.raises(ValueError, match="not on the strategies it wraps"):
            CombinedEarlyStopping([ValLossEarlyStopping(1, restore_best_weights=True)])


class TestRestore:
    @staticmethod
    def run(strategy: EarlyStopping, model: nn.Linear, losses: list[float]) -> None:
        for number, loss in enumerate(losses, start=1):
            with torch.no_grad():
                model.weight.fill_(float(number))
            strategy.update(make_report(number, val_loss=loss, model=model))

    def test_puts_the_best_weights_back(self) -> None:
        model = nn.Linear(1, 1)
        strategy = ValLossEarlyStopping(5, restore_best_weights=True)
        self.run(strategy, model, [1.0, 0.5, 0.7])

        strategy.restore(model)

        assert model.weight.item() == 2.0

    def test_snapshot_is_a_copy(self) -> None:
        model = nn.Linear(1, 1)
        strategy = ValLossEarlyStopping(5, restore_best_weights=True)
        self.run(strategy, model, [1.0])
        with torch.no_grad():
            model.weight.fill_(42.0)

        strategy.restore(model)

        assert model.weight.item() == 1.0

    def test_without_the_flag_leaves_the_model_alone(self) -> None:
        model = nn.Linear(1, 1)
        strategy = ValLossEarlyStopping(5)
        self.run(strategy, model, [1.0, 2.0])

        strategy.restore(model)

        assert model.weight.item() == 2.0

    def test_combined_restores_its_own_best(self) -> None:
        model = nn.Linear(1, 1)
        strategy = CombinedEarlyStopping(
            [ValLossEarlyStopping(5)], restore_best_weights=True
        )
        self.run(strategy, model, [1.0, 0.5, 0.9])

        strategy.restore(model)

        assert model.weight.item() == 2.0
