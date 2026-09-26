"""Streaming metrics accumulated batch by batch over one pass."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torchmetrics.text.rouge import ROUGEScore

from dl_roadmap.engine.trainer.batch import BatchParts
from dl_roadmap.engine.trainer.context import StepContext

MetricValue = float | Mapping[str, float]
"""One number, or several under sub-metric names for a metric reporting more."""

_ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")
"""ROUGE variants the ROUGE metrics report, besides their mean."""


class Metric(ABC):
    """Base interface for a metric aggregated over one pass's batches."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state before a training or validation pass."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        parts: BatchParts,
        predictions: torch.Tensor,
        ctx: StepContext,
    ) -> None:
        """Accumulate state from one batch.

        Args:
            parts: Batch tensors by role, already on the training device;
                targets are present.
            predictions: Model predictions for this batch, detached. Under
                mixed precision these are half precision; cast before any
                accumulation that needs full precision.
            ctx: Phase, interval and step this batch belongs to.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> MetricValue:
        """Return the aggregated value over all batches seen since `reset`.

        Returns:
            MetricValue: A single value, or a mapping of sub-metric name to
                value for a metric reporting several numbers at once.
        """
        raise NotImplementedError


def flatten_metric(name: str, value: MetricValue) -> dict[str, float]:
    """Flatten one metric's value into named scalars.

    Args:
        name: Name the metric is registered under.
        value: The metric's `compute` result.

    Returns:
        dict[str, float]: `name`, or "<name>_<sub-metric>" for a mapping,
            to its scalar value.
    """
    if isinstance(value, Mapping):
        return {f"{name}_{key}": float(sub) for key, sub in value.items()}

    return {name: float(value)}


class TokenAccuracy(Metric):
    """Score the share of non-padding target tokens predicted correctly."""

    def __init__(self, pad_id: int, vocab_dim: int = -1) -> None:
        """Initialize the metric with empty state.

        Args:
            pad_id: Token id used for padding, excluded from the counts.
            vocab_dim: Axis of the logits that holds the vocabulary: -1 for
                (batch, seq, vocab), 1 for (batch, vocab, seq).
        """
        self.pad_id = pad_id
        self.vocab_dim = vocab_dim
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._correct = 0
        self._total = 0

    def update(
        self,
        parts: BatchParts,
        predictions: torch.Tensor,
        _ctx: StepContext,
    ) -> None:
        """Accumulate correct and total non-padding token counts.

        Args:
            parts: Batch tensors by role; targets are token ids of shape
                (batch, seq).
            predictions: Logits with the vocabulary on `vocab_dim`, matching
                the targets once that axis is reduced.
            _ctx: Unused.

        Raises:
            ValueError: If the reduced logits do not match the targets'
                shape, e.g. because `vocab_dim` names the wrong axis.
        """
        targets = parts.require_targets()
        labels = predictions.argmax(dim=self.vocab_dim)

        if labels.shape != targets.shape:
            raise ValueError(
                f"Logits reduced over dim {self.vocab_dim} have shape "
                f"{tuple(labels.shape)}, but the targets {tuple(targets.shape)}."
            )

        mask = targets != self.pad_id
        self._correct += int((labels.eq(targets) & mask).sum().item())
        self._total += int(mask.sum().item())

    def compute(self) -> float:
        """Return the share of non-padding tokens predicted correctly.

        Returns:
            float: Correct over total non-padding tokens; 0.0 before any
                batch.
        """
        return self._correct / max(self._total, 1)


def _normalize(text: str) -> str:
    """Lowercase `text`, fold "ё" onto "е", and drop punctuation.

    Args:
        text: Raw decoded text.

    Returns:
        str: The normalized text. Unlike `ROUGEScore`'s default normalizer,
            this keeps non-ASCII letters, which the default strips to an
            empty string.
    """
    text = text.lower().replace("ё", "е")

    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _rouge_scorer() -> ROUGEScore:
    """Build the ROUGE accumulator both ROUGE metrics share.

    Returns:
        ROUGEScore: Scorer of `_ROUGE_KEYS` over `_normalize`d text, without
            stemming.
    """
    return ROUGEScore(rouge_keys=_ROUGE_KEYS, normalizer=_normalize, use_stemmer=False)


def _rouge_fmeasures(scorer: ROUGEScore) -> dict[str, float]:
    """Read each ROUGE F-measure off a scorer, and their mean.

    Args:
        scorer: Accumulator that has scored at least one pair of texts.

    Returns:
        dict[str, float]: "rouge1"/"rouge2"/"rougeL" to their F-measure, and
            "mean" to the average of the three.
    """
    scores = scorer.compute()
    values = {key: float(scores[f"{key}_fmeasure"].item()) for key in _ROUGE_KEYS}

    return {**values, "mean": sum(values.values()) / len(values)}


class RougeScore(Metric):
    """ROUGE-1/2/L F-measure over decoded predictions, plus their mean."""

    def __init__(
        self,
        decode: Callable[[list[list[int]]], list[str]],
        pad_id: int,
        on_train: bool = False,
        max_batches: int | None = None,
    ) -> None:
        """Initialize the metric with empty state.

        Args:
            decode: Maps a list of token id sequences to their texts, e.g.
                `sp.decode` of a SentencePiece processor.
            pad_id: Token id used for padding, excluded from both sequences.
            on_train: Whether to also score training batches. Off by default,
                since decoding and scoring every training batch is expensive.
            max_batches: Score at most this many batches per pass, sampling
                the first ones. None scores every batch.
        """
        self.decode = decode
        self.pad_id = pad_id
        self.on_train = on_train
        self.max_batches = max_batches

        self._rouge = _rouge_scorer()
        self._n_batches = 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._rouge.reset()
        self._n_batches = 0

    def update(
        self,
        parts: BatchParts,
        predictions: torch.Tensor,
        ctx: StepContext,
    ) -> None:
        """Decode one batch and accumulate its ROUGE scores.

        Args:
            parts: Batch tensors by role; targets are token ids of shape
                (batch, seq).
            predictions: Logits of shape (batch, seq, vocab).
            ctx: Phase, interval and step this batch belongs to; training
                batches are skipped unless `on_train` is set.

        Note:
            Predictions come from a teacher-forced pass, so the scores read
            higher than the ones free-running generation would produce.
        """
        if ctx.is_training and not self.on_train:
            return

        if self.max_batches is not None and self._n_batches >= self.max_batches:
            return

        labels = predictions.argmax(dim=-1).cpu()
        targets = parts.require_targets().cpu()
        keep = targets != self.pad_id

        predicted_ids = [row[mask].tolist() for row, mask in zip(labels, keep)]
        target_ids = [row[mask].tolist() for row, mask in zip(targets, keep)]

        self._rouge.update(self.decode(predicted_ids), self.decode(target_ids))
        self._n_batches += 1

    def compute(self) -> Mapping[str, float]:
        """Return each ROUGE F-measure and their mean.

        Returns:
            Mapping[str, float]: "rouge1"/"rouge2"/"rougeL" to their
                F-measure, and "mean" to the average of the three. Empty if
                no batch was scored since `reset`, e.g. on a training pass
                with `on_train` off.
        """
        if self._n_batches == 0:
            return {}

        return _rouge_fmeasures(self._rouge)


class GeneratedRougeScore(Metric):
    """ROUGE-1/2/L F-measure over free-running generations, plus their mean.

    Unlike `RougeScore`, this scores what the model generates from the source
    alone rather than the argmax of a teacher-forced pass, so it reflects
    actual generation quality.
    """

    def __init__(
        self,
        generate: Callable[..., torch.Tensor],
        decode: Callable[[list[list[int]]], list[str]],
        pad_id: int,
        max_examples: int | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the metric with empty state.

        Args:
            generate: Maps one source of shape ``1 x src_len`` to generated
                token ids, e.g. `model.generate`. Called one example at a
                time.
            decode: Maps a list of token id sequences to their texts, e.g.
                `sp.decode` of a SentencePiece processor.
            pad_id: Token id used for padding, excluded from sources and
                targets.
            max_examples: Score at most this many examples per pass, sampling
                the first ones. None scores every example. Unlike a batch
                limit, this stays fixed when the batch size changes.
            generation_kwargs: Keyword arguments passed to `generate` for
                every example.

        Raises:
            ValueError: If `max_examples` is below 1.
        """
        if max_examples is not None and max_examples < 1:
            raise ValueError(f"max_examples must be >= 1, got {max_examples}.")

        self.generate = generate
        self.decode = decode
        self.pad_id = pad_id
        self.max_examples = max_examples
        self.generation_kwargs = dict(generation_kwargs or {})

        self._rouge = _rouge_scorer()
        self._n_examples = 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._rouge.reset()
        self._n_examples = 0

    @torch.no_grad()
    def update(
        self,
        parts: BatchParts,
        _predictions: torch.Tensor,
        ctx: StepContext,
    ) -> None:
        """Generate from each source and accumulate the ROUGE scores.

        Args:
            parts: Batch tensors by role; inputs and targets are token ids of
                shape (batch, seq).
            _predictions: Unused.
            ctx: Phase, interval and step this batch belongs to; training
                batches are skipped.

        Raises:
            ValueError: If `generate` returns neither of the accepted shapes.
        """
        if ctx.is_training:
            return

        inputs, targets = parts.inputs, parts.require_targets()
        remaining = inputs.shape[0]
        if self.max_examples is not None:
            remaining = min(remaining, self.max_examples - self._n_examples)

        if remaining <= 0:
            return

        generated_ids: list[list[int]] = []
        target_ids: list[list[int]] = []
        for source, target in zip(inputs[:remaining], targets[:remaining]):
            generated = self.generate(
                source[source != self.pad_id].unsqueeze(0), **self.generation_kwargs
            )
            if generated.ndim == 2:  # noqa: PLR2004
                generated = generated[0]

            if generated.ndim != 1:
                raise ValueError(
                    "generate must return token ids shaped `seq_len` or "
                    f"`1 x seq_len`, got {tuple(generated.shape)}"
                )

            generated_ids.append(generated.cpu().tolist())
            target_ids.append(target[target != self.pad_id].cpu().tolist())

        self._rouge.update(self.decode(generated_ids), self.decode(target_ids))
        self._n_examples += remaining

    def compute(self) -> Mapping[str, float]:
        """Return each ROUGE F-measure and their mean.

        Returns:
            Mapping[str, float]: "rouge1"/"rouge2"/"rougeL" to their
                F-measure, and "mean" to the average of the three. Empty if
                no example was scored since `reset`, e.g. on a training pass.
        """
        if self._n_examples == 0:
            return {}

        return _rouge_fmeasures(self._rouge)
