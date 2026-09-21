"""Streaming metrics accumulated batch by batch during a training epoch."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torchmetrics.text.rouge import ROUGEScore

from dl_roadmap.engine.trainer.context import StepContext

MetricValue = float | Mapping[str, float]


class Metric(ABC):
    """Base interface for a metric aggregated over one epoch's batches."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state, e.g. at the start of an epoch."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        extras: list[torch.Tensor],
        predictions: torch.Tensor,
        ctx: StepContext,
    ) -> None:
        """Accumulate state from one batch.

        Args:
            inputs: Batch inputs, already moved to the training device.
            targets: Batch targets, already moved to the training device.
            extras: Any batch elements beyond inputs/targets, already moved
                to the training device.
            predictions: Model predictions for this batch, detached. Under
                mixed precision these are half precision; cast before any
                accumulation that needs full precision.
            ctx: Phase, epoch and step this batch belongs to.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> MetricValue:
        """Return the aggregated value over all batches seen since `reset`.

        Returns:
            A single value, or a mapping of sub-metric name to value for a
            metric reporting several numbers at once.
        """
        raise NotImplementedError


def flatten_metric(name: str, value: MetricValue) -> dict[str, float]:
    """Flatten one metric's value into named scalars.

    Args:
        name: Name the metric is registered under.
        value: The metric's `compute` result.

    Returns:
        Mapping of `name` (or "<name>_<sub-metric>") to scalar value.
    """
    if isinstance(value, Mapping):
        return {f"{name}_{key}": float(sub) for key, sub in value.items()}

    return {name: float(value)}


class TokenAccuracy(Metric):
    """Share of non-padding target tokens predicted correctly."""

    def __init__(self, pad_id: int) -> None:
        """Initialize the metric with empty state.

        Args:
            pad_id: Token id used for padding, excluded from the counts.
        """
        self.pad_id = pad_id
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._correct = 0
        self._total = 0

    def update(
        self,
        _inputs: torch.Tensor,
        targets: torch.Tensor,
        _extras: list[torch.Tensor],
        predictions: torch.Tensor,
        _ctx: StepContext,
    ) -> None:
        """Accumulate correct and total non-padding token counts.

        Args:
            _inputs: Unused.
            targets: Batch target token ids, shape (batch, seq).
            _extras: Unused.
            predictions: Logits of shape (batch, seq, vocab) or (batch,
                vocab, seq), matching `targets` once the vocab axis is
                reduced.
            _ctx: Unused.
        """
        labels = predictions.argmax(dim=-1)
        if labels.shape != targets.shape:
            labels = predictions.argmax(dim=1)

        mask = targets != self.pad_id
        self._correct += int((labels.eq(targets) & mask).sum().item())
        self._total += int(mask.sum().item())

    def compute(self) -> float:
        """Return the share of non-padding tokens predicted correctly."""
        return self._correct / max(self._total, 1)


def _normalize(text: str) -> str:
    """Lowercase `text`, fold "ё" onto "е", and drop punctuation.

    Args:
        text: Raw decoded text.

    Returns:
        The normalized text. Unlike `ROUGEScore`'s default normalizer, this
        keeps non-ASCII letters, which the default strips to an empty string.
    """
    text = text.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


class RougeScore(Metric):
    """ROUGE-1/2/L F-measure over decoded predictions, plus their mean."""

    ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")

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

        self._rouge = ROUGEScore(
            rouge_keys=self.ROUGE_KEYS,
            normalizer=_normalize,
            use_stemmer=False,
        )
        self._n_batches = 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._rouge.reset()
        self._n_batches = 0

    def update(
        self,
        _inputs: torch.Tensor,
        targets: torch.Tensor,
        _extras: list[torch.Tensor],
        predictions: torch.Tensor,
        ctx: StepContext,
    ) -> None:
        """Decode one batch and accumulate its ROUGE scores.

        Args:
            _inputs: Unused.
            targets: Batch target token ids, shape (batch, seq).
            _extras: Unused.
            predictions: Logits of shape (batch, seq, vocab).
            ctx: Phase, epoch and step this batch belongs to; training
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
        targets = targets.cpu()
        keep = targets != self.pad_id

        predicted_ids = [row[mask].tolist() for row, mask in zip(labels, keep)]
        target_ids = [row[mask].tolist() for row, mask in zip(targets, keep)]

        self._rouge.update(self.decode(predicted_ids), self.decode(target_ids))
        self._n_batches += 1

    def compute(self) -> Mapping[str, float]:
        """Return each ROUGE F-measure and their mean.

        Returns:
            Mapping of "rouge1"/"rouge2"/"rougeL" to their F-measure, and
            "mean" to the average of the three. Empty if no batch was scored
            since `reset`, e.g. on a training pass with `on_train` off.
        """
        if self._n_batches == 0:
            return {}

        scores = self._rouge.compute()
        values = {
            key: float(scores[f"{key}_fmeasure"].item()) for key in self.ROUGE_KEYS
        }

        return {**values, "mean": sum(values.values()) / len(values)}


class GeneratedRougeScore(Metric):
    """ROUGE over free-running autoregressive model generations.

    Unlike :class:`RougeScore`, this metric never scores argmax tokens from
    the teacher-forced validation forward pass.  It calls the model's
    generation function using only the source sequence, so it is suitable as
    an Optuna objective for sequence generation quality.

    Generation is intentionally performed one example at a time because the
    current summarizer's ``generate`` method accepts a batch size of one.
    ``max_examples`` keeps this relatively expensive metric practical during
    tuning and, unlike a batch limit, evaluates the same number of examples
    when the DataLoader batch size changes.
    """

    ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")
    SEQUENCE_NDIM = 1
    BATCHED_SEQUENCE_NDIM = 2

    def __init__(
        self,
        generate: Callable[..., torch.Tensor],
        decode: Callable[[list[list[int]]], list[str]],
        pad_id: int,
        max_examples: int | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a free-running ROUGE accumulator.

        Args:
            generate: Callable receiving one source tensor shaped
                ``1 x src_len`` and returning generated token ids. Typically
                ``model.generate``.
            decode: Batched token-id decoder, e.g. ``sp.decode``.
            pad_id: Padding token id, removed from sources and references.
            max_examples: Maximum validation examples scored per epoch. None
                evaluates the complete validation pass.
            generation_kwargs: Fixed keyword arguments passed to ``generate``
                for every example. These must remain identical across trials.
        """
        if max_examples is not None and max_examples < 1:
            raise ValueError("max_examples must be >= 1 or None")

        self.generate = generate
        self.decode = decode
        self.pad_id = pad_id
        self.max_examples = max_examples
        self.generation_kwargs = dict(generation_kwargs or {})
        self._rouge = ROUGEScore(
            rouge_keys=self.ROUGE_KEYS,
            normalizer=_normalize,
            use_stemmer=False,
        )
        self._n_examples = 0

    def reset(self) -> None:
        """Clear accumulated generations and scores."""
        self._rouge.reset()
        self._n_examples = 0

    @torch.no_grad()
    def update(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        _extras: list[torch.Tensor],
        _predictions: torch.Tensor,
        ctx: StepContext,
    ) -> None:
        """Generate summaries from sources and accumulate validation ROUGE."""
        if ctx.is_training:
            return

        remaining = (
            inputs.shape[0]
            if self.max_examples is None
            else min(inputs.shape[0], self.max_examples - self._n_examples)
        )
        if remaining <= 0:
            return

        generated_ids: list[list[int]] = []
        target_ids: list[list[int]] = []

        for source, target in zip(inputs[:remaining], targets[:remaining]):
            source_input = source[source != self.pad_id].unsqueeze(0)
            generated = self.generate(source_input, **self.generation_kwargs)

            if generated.ndim == self.BATCHED_SEQUENCE_NDIM:
                generated = generated[0]
            if generated.ndim != self.SEQUENCE_NDIM:
                raise ValueError(
                    "generate must return token ids shaped seq_len or "
                    f"1 x seq_len, got {tuple(generated.shape)}"
                )

            generated_ids.append(generated.detach().cpu().tolist())
            target_ids.append(target[target != self.pad_id].detach().cpu().tolist())

        self._rouge.update(
            self.decode(generated_ids),
            self.decode(target_ids),
        )
        self._n_examples += remaining

    def compute(self) -> Mapping[str, float]:
        """Return ROUGE-1/2/L F-measure and their arithmetic mean."""
        if self._n_examples == 0:
            return {}

        scores = self._rouge.compute()
        values = {
            key: float(scores[f"{key}_fmeasure"].item()) for key in self.ROUGE_KEYS
        }
        return {**values, "mean": sum(values.values()) / len(values)}
