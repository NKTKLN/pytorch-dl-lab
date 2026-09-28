"""EpochSchedule, StepSchedule and TokenSchedule."""

from typing import cast

import pytest
import torch

from dl_roadmap.engine.trainer import EpochSchedule, Optimization
from dl_roadmap.engine.trainer.schedule import StepSchedule, TokenSchedule


class StepCounter:
    """Stands in for an engine; only `steps` is read."""

    def __init__(self) -> None:
        self.steps = 0


def batches(count: int) -> list[tuple[torch.Tensor]]:
    return [(torch.full((1,), float(i)),) for i in range(count)]


class TestEpochSchedule:
    def test_rejects_zero_epochs(self) -> None:
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            EpochSchedule(0)

    def test_yields_one_interval_per_epoch_from_start(self) -> None:
        loader = batches(3)
        intervals = list(EpochSchedule[tuple[torch.Tensor]](4).intervals(loader, 2))

        assert [i.number for i in intervals] == [2, 3, 4]
        assert all(i.batches is loader and i.total == 3 for i in intervals)
        assert all(i.position is None and i.axis_unit == "epoch" for i in intervals)

    def test_starts_at_the_first_epoch_counting_batches(self) -> None:
        loader = batches(2)
        intervals = list(EpochSchedule[tuple[torch.Tensor]](2).intervals(loader))

        assert [i.number for i in intervals] == [1, 2]
        assert [intervals[0].units(batch) for batch in loader] == [1, 1]

    @pytest.mark.parametrize("start", [0, 5])
    def test_rejects_start_outside_the_run(self, start: int) -> None:
        with pytest.raises(ValueError, match=r"start must be in 1\.\.4"):
            list(EpochSchedule[tuple[torch.Tensor]](4).intervals(batches(1), start))


class TestStepSchedule:
    @pytest.mark.parametrize(("max_steps", "eval_every"), [(0, 1), (1, 0)])
    def test_rejects_empty_budgets(self, max_steps: int, eval_every: int) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            StepSchedule(cast(Optimization, StepCounter()), max_steps, eval_every)

    def test_accepts_a_single_step(self) -> None:
        assert len(StepSchedule(cast(Optimization, StepCounter()), 1, 1)) == 1

    def test_windows_cover_the_budget_over_a_cycled_loader(self) -> None:
        counter = StepCounter()
        schedule = StepSchedule[tuple[torch.Tensor]](
            cast(Optimization, counter), max_steps=5, eval_every=2
        )
        seen: list[tuple[int, int | None, int | None, int]] = []

        assert len(schedule) == 3

        for interval in schedule.intervals(batches(3)):
            consumed = 0
            for _ in interval.batches:
                counter.steps += 1
                consumed += 1
            seen.append((interval.number, interval.position, interval.total, consumed))

        assert seen == [(1, 2, 2, 2), (2, 4, 2, 2), (3, 5, 1, 1)]
        assert counter.steps == 5

    def test_accumulating_batches_count_zero_steps(self) -> None:
        counter = StepCounter()
        schedule = StepSchedule[tuple[torch.Tensor]](
            cast(Optimization, counter), max_steps=2, eval_every=2
        )
        (interval,) = schedule.intervals(batches(10))
        advances = []

        for index, batch in enumerate(interval.batches):
            if index % 2:
                counter.steps += 1
            advances.append(interval.units(batch))

        assert advances == [0, 1, 0, 1]

    def test_resumes_from_a_later_window(self) -> None:
        schedule = StepSchedule[tuple[torch.Tensor]](
            cast(Optimization, StepCounter()), max_steps=5, eval_every=2
        )
        positions = [i.position for i in schedule.intervals(batches(1), start=2)]

        assert positions == [4, 5]


class TestTokenSchedule:
    def test_windows_close_once_their_tokens_are_spent(self) -> None:
        schedule = TokenSchedule[tuple[torch.Tensor]](budget=10, eval_every=4, pad_id=0)
        loader = [(torch.tensor([[1, 2, 3, 0]]),)]
        seen = []

        for interval in schedule.intervals(loader):
            tokens = [interval.units(batch) for batch in interval.batches]
            seen.append((interval.number, interval.position, tokens))

        assert seen == [(1, 4, [3, 3]), (2, 8, [3, 3]), (3, 10, [3])]

    @pytest.mark.parametrize(
        ("budget", "eval_every", "windows"), [(10, 4, 3), (8, 4, 2), (9, 4, 3)]
    )
    def test_length_counts_a_partial_last_window(
        self, budget: int, eval_every: int, windows: int
    ) -> None:
        schedule = TokenSchedule[tuple[torch.Tensor]](budget, eval_every)

        assert len(schedule) == windows

    def test_a_one_token_last_window_is_still_run(self) -> None:
        schedule = TokenSchedule[tuple[torch.Tensor]](budget=9, eval_every=4)
        intervals = list(schedule.intervals([(torch.zeros(1, 1),)]))

        assert [(i.position, i.total, i.unit) for i in intervals] == [
            (4, 4, "token"),
            (8, 4, "token"),
            (9, 1, "token"),
        ]

    def test_accepts_a_single_token(self) -> None:
        assert len(TokenSchedule[tuple[torch.Tensor]](1, 1)) == 1

    def test_counts_every_position_without_a_pad_id(self) -> None:
        schedule = TokenSchedule[tuple[torch.Tensor]](budget=8, eval_every=8)
        (interval,) = schedule.intervals([(torch.zeros(2, 4),)])

        assert [interval.units(b) for b in interval.batches] == [8]

    @pytest.mark.parametrize(("budget", "eval_every"), [(0, 1), (1, 0)])
    def test_rejects_empty_budgets(self, budget: int, eval_every: int) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            TokenSchedule[tuple[torch.Tensor]](budget, eval_every)
