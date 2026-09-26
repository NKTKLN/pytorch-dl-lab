"""The optimizer step: accumulation, normalization, clipping, AMP and state."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from dl_roadmap.engine.trainer import (
    NoOptimization,
    OptimizationConfig,
    OptimizationEngine,
    Warmup,
)

CPU = torch.device("cpu")


def engine(
    model: nn.Module,
    lr: float = 0.1,
    momentum: float = 0.0,
    **config: object,
) -> OptimizationEngine:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    settings = OptimizationConfig(amp="off")
    for name, value in config.items():
        setattr(settings, name, value)
    settings.__post_init__()

    return OptimizationEngine(optimizer, config=settings).prepare(CPU)


def linear(seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(3, 1)


class TestConfig:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("grad_clip_norm", 0.0),
            ("accumulation_steps", 0),
            ("grad_normalizer", "tokens"),
            ("amp", "fp8"),
        ],
    )
    def test_rejects_invalid_settings(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field):
            OptimizationConfig(**{field: value})  # type: ignore[arg-type]


class TestNoOptimization:
    def test_cannot_train(self) -> None:
        stub = NoOptimization()

        with pytest.raises(RuntimeError, match="without an OptimizationEngine"):
            stub.backward(torch.tensor(1.0))

        assert not stub.can_step
        assert (stub.step_if_ready(), stub.flush(), stub.state_dict()) == (
            False,
            False,
            {},
        )


def test_engine_must_be_prepared() -> None:
    model = linear()
    raw = OptimizationEngine(torch.optim.SGD(model.parameters(), lr=0.1))

    with pytest.raises(RuntimeError, match="not prepared"):
        raw.backward(model(torch.ones(1, 3)).sum())


class TestAccumulation:
    def test_mean_loss_windows_match_one_big_batch(self) -> None:
        x, y = torch.randn(8, 3), torch.randn(8, 1)
        whole, split = linear(), linear()
        big = engine(whole)
        small = engine(split, accumulation_steps=4)

        big.backward(nn.functional.mse_loss(whole(x), y))
        big.step_if_ready()

        stepped = []
        for start in range(0, 8, 2):
            xb, yb = x[start : start + 2], y[start : start + 2]
            small.backward(nn.functional.mse_loss(split(xb), yb))
            stepped.append(small.step_if_ready())

        assert stepped == [False, False, False, True]
        assert small.steps == big.steps == 1
        torch.testing.assert_close(split.weight, whole.weight)

    def test_summed_loss_weighted_by_tokens_matches_one_big_batch(self) -> None:
        pad = 0
        targets = torch.tensor([[1, 2, pad, pad], [3, 1, 2, 3]])
        inputs = torch.randn(2, 4, 5)
        whole, split = nn.Linear(5, 4), nn.Linear(5, 4)
        split.load_state_dict(whole.state_dict())
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad, reduction="sum")

        def loss(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            value: torch.Tensor = loss_fn(model(x).reshape(-1, 4), t.reshape(-1))
            return value

        big = engine(whole, grad_normalizer="loss_weights")
        big.backward(loss(whole, inputs, targets), float((targets != pad).sum()))
        big.step_if_ready()

        small = engine(split, accumulation_steps=2, grad_normalizer="loss_weights")
        for row in range(2):
            x, t = inputs[row : row + 1], targets[row : row + 1]
            small.backward(loss(split, x, t), float((t != pad).sum()))
            small.step_if_ready()

        torch.testing.assert_close(split.weight, whole.weight)

    def test_flush_steps_a_partial_window_once(self) -> None:
        model = linear()
        eng = engine(model, accumulation_steps=4)
        eng.backward(model(torch.ones(1, 3)).sum())

        assert (eng.flush(), eng.flush(), eng.steps) == (True, False, 1)

    def test_weightless_window_is_skipped(self) -> None:
        model = linear()
        before = model.weight.detach().clone()
        eng = engine(model)
        eng.backward(model(torch.ones(1, 3)).sum(), batch_weight=0.0)

        assert eng.step_if_ready()
        assert eng.steps == 0
        torch.testing.assert_close(model.weight, before)

    def test_gradients_do_not_leak_between_windows(self) -> None:
        model = linear()
        eng = engine(model, lr=0.0)
        eng.backward(model(torch.ones(1, 3)).sum())
        eng.step_if_ready()
        eng.backward(model(torch.ones(1, 3)).sum())

        torch.testing.assert_close(model.weight.grad, torch.ones(1, 3))


def test_clipping_bounds_the_update() -> None:
    model = linear()
    before = model.weight.detach().clone()
    eng = engine(model, lr=1.0, grad_clip_norm=0.5)
    eng.backward(1000 * model(torch.ones(1, 3)).sum())
    eng.step_if_ready()

    update = torch.cat([(model.weight - before).flatten()])

    assert update.norm() <= 0.5 + 1e-6


def test_schedule_advances_only_on_real_steps() -> None:
    model = linear()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    eng = OptimizationEngine(
        optimizer,
        Warmup(optimizer, 4),
        OptimizationConfig(amp="off", accumulation_steps=2),
    ).prepare(CPU)

    for _ in range(3):
        eng.backward(model(torch.ones(1, 3)).sum())
        eng.step_if_ready()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.4375)


def test_state_round_trips_with_momentum() -> None:
    first_model, second_model = linear(), linear()
    first = engine(first_model, momentum=0.9)
    for _ in range(2):
        first.backward(first_model(torch.ones(1, 3)).sum())
        first.step_if_ready()

    second = engine(second_model, momentum=0.9)
    second_model.load_state_dict(first_model.state_dict())
    second.load_state_dict(deepcopy(first.state_dict()))  # as a file load would

    for eng, model in ((first, first_model), (second, second_model)):
        eng.backward(model(torch.ones(1, 3)).sum())
        eng.step_if_ready()

    assert second.steps == first.steps == 3
    torch.testing.assert_close(second_model.weight, first_model.weight)


class TestAmp:
    @pytest.mark.parametrize(
        ("amp", "expected"),
        [
            ("auto", (False, torch.float16)),
            ("off", (False, torch.float16)),
            ("bf16", (True, torch.bfloat16)),
            ("fp16", (True, torch.float16)),
        ],
    )
    def test_resolves_on_cpu(
        self, amp: str, expected: tuple[bool, torch.dtype]
    ) -> None:
        eng = engine(linear(), amp=amp)

        assert (eng.amp_enabled, eng.amp_dtype) == expected

    @pytest.mark.parametrize("amp", ["bf16", "fp16"])
    def test_trains_under_cpu_autocast(self, amp: str) -> None:
        model = linear()
        before = model.weight.detach().clone()
        eng = engine(model, amp=amp)

        for _ in range(5):  # fp16 skips steps while the scaler backs off
            with eng.autocast():
                loss = model(torch.ones(4, 3)).float().pow(2).mean()
            eng.backward(loss)
            eng.step_if_ready()

        assert eng.steps >= 1
        assert not torch.equal(model.weight, before)

    def test_prepare_is_idempotent(self) -> None:
        eng = engine(linear())
        scaler = eng.scaler

        assert eng.prepare(CPU).scaler is scaler
