"""Learning-rate schedules: cadence, monitoring, warmup and resuming."""

import math

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from dl_roadmap.engine.trainer import TorchLRSchedule, Warmup
from dl_roadmap.engine.trainer.lr_schedulers import as_lr_schedule

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detected call of `lr_scheduler.step\\(\\)` before:UserWarning"
)


def optimizer(lr: float = 1.0) -> torch.optim.SGD:
    return torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=lr)


def lr_of(opt: torch.optim.Optimizer) -> float:
    return float(opt.param_groups[0]["lr"])


class TestTorchLRSchedule:
    def test_interval_cadence_ignores_steps(self) -> None:
        opt = optimizer()
        schedule = TorchLRSchedule(StepLR(opt, step_size=1, gamma=0.5))
        schedule.after_step()

        assert lr_of(opt) == 1.0

        schedule.after_interval({})

        assert lr_of(opt) == 0.5

    def test_step_cadence_ignores_intervals(self) -> None:
        opt = optimizer()
        schedule = TorchLRSchedule(StepLR(opt, 1, 0.5), every="step")
        schedule.after_interval({})

        assert lr_of(opt) == 1.0

        schedule.after_step()

        assert lr_of(opt) == 0.5

    def test_plateau_steps_on_the_monitored_value(self) -> None:
        opt = optimizer()
        schedule = TorchLRSchedule(
            ReduceLROnPlateau(opt, mode="max", patience=0), monitor="val_acc"
        )
        schedule.after_interval({"val_acc": 0.5, "val_loss": 1.0})
        schedule.after_interval({"val_acc": 0.4, "val_loss": 0.1})

        assert lr_of(opt) == pytest.approx(0.1)

    def test_plateau_without_its_value_raises(self) -> None:
        schedule = TorchLRSchedule(ReduceLROnPlateau(optimizer()))

        with pytest.raises(ValueError, match="monitors 'val_loss'"):
            schedule.after_interval({"train_loss": 1.0})

    def test_plateau_cannot_step_per_optimizer_step(self) -> None:
        with pytest.raises(ValueError, match="only advance every"):
            TorchLRSchedule(ReduceLROnPlateau(optimizer()), every="step")

    def test_state_round_trips(self) -> None:
        first = TorchLRSchedule(StepLR(optimizer(), 1, 0.5))
        first.after_interval({})
        second = TorchLRSchedule(StepLR(optimizer(), 1, 0.5))
        second.load_state_dict(first.state_dict())

        assert second.state_dict() == first.state_dict()


def test_as_lr_schedule_wraps_bare_schedulers_only() -> None:
    schedule = TorchLRSchedule(StepLR(optimizer(), 1))

    assert as_lr_schedule(schedule) is schedule
    assert isinstance(as_lr_schedule(StepLR(optimizer(), 1)), TorchLRSchedule)


class TestWarmup:
    def test_ramps_per_step_then_hands_over(self) -> None:
        opt = optimizer()
        then = TorchLRSchedule(StepLR(opt, 1, 0.5), every="step")
        schedule = Warmup(opt, 4, then=then)
        lrs = [lr_of(opt)]

        for _ in range(6):
            schedule.after_step()
            lrs.append(lr_of(opt))

        assert lrs == pytest.approx([0.25, 0.4375, 0.625, 0.8125, 1.0, 0.5, 0.25])

    def test_interval_ramp_ignores_steps(self) -> None:
        opt = optimizer()
        schedule = Warmup(opt, 2, every="interval")
        schedule.after_step()

        assert lr_of(opt) == 0.5

        schedule.after_interval({})
        schedule.after_interval({})

        assert lr_of(opt) == 1.0
        assert not schedule.in_warmup

    def test_holds_back_the_next_schedule_during_the_ramp(self) -> None:
        opt = optimizer()
        schedule = Warmup(opt, 2, then=ReduceLROnPlateau(opt, patience=0))
        schedule.after_interval({})

        schedule.after_step()
        schedule.after_step()

        with pytest.raises(ValueError, match="monitors 'val_loss'"):
            schedule.after_interval({})

    def test_cosine_after_the_ramp_follows_its_closed_form(self) -> None:
        opt = optimizer()
        cosine = CosineAnnealingLR(opt, T_max=10)
        schedule = Warmup(opt, 3, then=TorchLRSchedule(cosine, every="step"))

        for _ in range(3):
            schedule.after_step()

        for k in range(1, 11):
            schedule.after_step()

            assert lr_of(opt) == pytest.approx((1 + math.cos(math.pi * k / 10)) / 2)

    def test_resumes_where_it_stopped(self) -> None:
        def build() -> tuple[torch.optim.SGD, Warmup]:
            opt = optimizer()
            then = TorchLRSchedule(StepLR(opt, 1, 0.5), every="step")
            return opt, Warmup(opt, 3, then=then)

        opt, schedule = build()
        for _ in range(2):
            schedule.after_step()

        resumed_opt, resumed = build()
        resumed_opt.load_state_dict(opt.state_dict())
        resumed.load_state_dict(schedule.state_dict())

        for _ in range(3):
            schedule.after_step()
            resumed.after_step()

            assert lr_of(resumed_opt) == lr_of(opt)

    def test_rejects_an_empty_ramp(self) -> None:
        with pytest.raises(ValueError, match="length must be >= 1"):
            Warmup(optimizer(), 0)
