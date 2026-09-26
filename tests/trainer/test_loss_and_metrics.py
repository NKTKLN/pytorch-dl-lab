"""Loss trackers, online metrics and the batch roles they read."""

from collections.abc import Callable

import pytest
import torch

from dl_roadmap.engine.trainer import (
    BatchParts,
    GeneratedRougeScore,
    MeanLossTracker,
    PerTokenLossTracker,
    RougeScore,
    StepContext,
    TokenAccuracy,
)
from dl_roadmap.engine.trainer.online_metrics import _normalize, flatten_metric

PAD = 0
VAL = StepContext(phase="val")
TRAIN = StepContext(phase="train")
WORDS = ["alpha", "beta", "gamma", "delta"]


def decode(sequences: list[list[int]]) -> list[str]:
    return [" ".join(WORDS[token - 1] for token in seq) for seq in sequences]


def one_hot(labels: torch.Tensor, vocab: int = 5) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, vocab).float()


def test_require_targets_rejects_an_inputs_only_batch() -> None:
    with pytest.raises(ValueError, match="no targets"):
        BatchParts(torch.zeros(1), None).require_targets()


class TestLossTrackers:
    def test_mean_averages_batches(self) -> None:
        tracker = MeanLossTracker()
        parts = BatchParts(torch.zeros(1), torch.zeros(1))

        assert tracker.compute() == 0.0

        for loss in (1.0, 3.0):
            tracker.update(torch.tensor(loss), parts, torch.zeros(1), VAL)

        assert (tracker.compute(), tracker.batch_weight(parts)) == (2.0, 1.0)

        tracker.reset()

        assert tracker.compute() == 0.0

    def test_per_token_divides_the_summed_loss_by_real_tokens(self) -> None:
        tracker = PerTokenLossTracker(PAD)
        first = BatchParts(torch.zeros(1), torch.tensor([[1, 2, PAD]]))
        second = BatchParts(torch.zeros(1), torch.tensor([[3, 4, 1]]))
        tracker.update(torch.tensor(4.0), first, torch.zeros(1), VAL)
        tracker.update(torch.tensor(6.0), second, torch.zeros(1), VAL)

        assert tracker.compute() == 2.0
        assert (tracker.batch_weight(first), tracker.batch_weight(second)) == (2, 3)


class TestFlattenMetric:
    def test_scalar_keeps_the_name(self) -> None:
        assert flatten_metric("acc", 0.5) == {"acc": 0.5}

    def test_mapping_prefixes_each_entry(self) -> None:
        assert flatten_metric("rouge", {"mean": 1, "rouge1": 0.5}) == {
            "rouge_mean": 1.0,
            "rouge_rouge1": 0.5,
        }


class TestTokenAccuracy:
    TARGETS = torch.tensor([[1, 2, 3, PAD]])
    LABELS = torch.tensor([[1, 2, 4, 4]])

    def test_scores_only_real_tokens(self) -> None:
        metric = TokenAccuracy(PAD)
        metric.reset()
        metric.update(BatchParts(self.TARGETS, self.TARGETS), one_hot(self.LABELS), VAL)

        assert metric.compute() == pytest.approx(2 / 3)

    def test_accepts_the_vocab_axis_in_the_middle(self) -> None:
        metric = TokenAccuracy(PAD, vocab_dim=1)
        metric.reset()
        logits = one_hot(self.LABELS).transpose(1, 2)
        metric.update(BatchParts(self.TARGETS, self.TARGETS), logits, VAL)

        assert metric.compute() == pytest.approx(2 / 3)

    def test_vocab_axis_in_the_middle_when_vocab_equals_length(self) -> None:
        metric = TokenAccuracy(PAD, vocab_dim=1)
        metric.reset()
        logits = one_hot(torch.tensor([[1, 2, 0, 0]]), vocab=4).transpose(1, 2)
        metric.update(BatchParts(self.TARGETS, self.TARGETS), logits, VAL)

        assert metric.compute() == pytest.approx(2 / 3)

    def test_rejects_logits_reduced_over_the_wrong_axis(self) -> None:
        metric = TokenAccuracy(PAD)
        logits = one_hot(self.LABELS).transpose(1, 2)

        with pytest.raises(ValueError, match="Logits reduced over dim -1"):
            metric.update(BatchParts(self.TARGETS, self.TARGETS), logits, VAL)

    def test_is_zero_before_any_batch(self) -> None:
        metric = TokenAccuracy(PAD)
        metric.reset()

        assert metric.compute() == 0.0


def test_normalize_keeps_cyrillic_and_folds_yo() -> None:
    assert _normalize("  Ёлка,   ДОМ!\n") == "елка дом"


class TestRougeScore:
    TOKENS = torch.tensor([[1, 2, 3, PAD], [2, 3, 4, 1]])

    def update(self, metric: RougeScore, ctx: StepContext = VAL) -> None:
        metric.update(BatchParts(self.TOKENS, self.TOKENS), one_hot(self.TOKENS), ctx)

    def test_perfect_predictions_score_one(self) -> None:
        metric = RougeScore(decode, PAD)
        metric.reset()
        self.update(metric)

        assert metric.compute() == pytest.approx(
            {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0, "mean": 1.0}
        )

    def test_skips_training_batches_unless_asked(self) -> None:
        metric = RougeScore(decode, PAD)
        metric.reset()
        self.update(metric, TRAIN)

        assert metric.compute() == {}

        scoring = RougeScore(decode, PAD, on_train=True)
        scoring.reset()
        self.update(scoring, TRAIN)

        assert scoring.compute()

    def test_caps_the_scored_batches(self) -> None:
        metric = RougeScore(decode, PAD, max_batches=1)
        metric.reset()
        self.update(metric)
        wrong = torch.tensor([[4, 4, 4, 4], [4, 4, 4, 4]])
        metric.update(BatchParts(self.TOKENS, self.TOKENS), one_hot(wrong), VAL)

        assert metric.compute()["mean"] == pytest.approx(1.0)


class TestGeneratedRougeScore:
    SOURCES = torch.tensor([[1, 2, PAD], [3, 4, PAD], [1, 3, PAD]])

    @staticmethod
    def echo(calls: list[torch.Tensor]) -> Callable[..., torch.Tensor]:
        def generate(source: torch.Tensor, **kwargs: object) -> torch.Tensor:
            calls.append(source)
            assert kwargs == {"beam": 2}
            return source

        return generate

    def test_generates_from_unpadded_sources_up_to_max_examples(self) -> None:
        calls: list[torch.Tensor] = []
        metric = GeneratedRougeScore(
            self.echo(calls), decode, PAD, max_examples=4, generation_kwargs={"beam": 2}
        )
        metric.reset()
        parts = BatchParts(self.SOURCES, self.SOURCES)
        metric.update(parts, torch.zeros(1), VAL)
        metric.update(parts, torch.zeros(1), VAL)

        assert len(calls) == 4
        assert all(call.shape == (1, 2) for call in calls)
        assert metric.compute()["mean"] == pytest.approx(1.0)

    def test_skips_training_and_empty_passes(self) -> None:
        metric = GeneratedRougeScore(self.echo([]), decode, PAD)
        metric.reset()
        metric.update(BatchParts(self.SOURCES, self.SOURCES), torch.zeros(1), TRAIN)

        assert metric.compute() == {}

    def test_accepts_flat_generations(self) -> None:
        metric = GeneratedRougeScore(lambda source: source[0], decode, PAD)
        metric.reset()
        metric.update(BatchParts(self.SOURCES, self.SOURCES), torch.zeros(1), VAL)

        assert metric.compute()["mean"] == pytest.approx(1.0)

    def test_rejects_other_generation_shapes(self) -> None:
        metric = GeneratedRougeScore(lambda s: s.unsqueeze(0), decode, PAD)
        metric.reset()

        with pytest.raises(ValueError, match="generate must return"):
            metric.update(BatchParts(self.SOURCES, self.SOURCES), torch.zeros(1), VAL)

    def test_rejects_max_examples_below_one(self) -> None:
        with pytest.raises(ValueError, match="max_examples must be >= 1"):
            GeneratedRougeScore(lambda s: s, decode, PAD, max_examples=0)
