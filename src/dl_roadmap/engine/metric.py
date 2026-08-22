"""Streaming metrics accumulated batch by batch during a training epoch."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import torch
from torchmetrics.text.rouge import ROUGEScore

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
        train: bool,
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
            train: Whether this batch was part of a training pass.
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
        _train: bool,
    ) -> None:
        """Accumulate correct and total non-padding token counts.

        Args:
            _inputs: Unused.
            targets: Batch target token ids, shape (batch, seq).
            _extras: Unused.
            predictions: Logits of shape (batch, seq, vocab) or (batch,
                vocab, seq), matching `targets` once the vocab axis is
                reduced.
            _train: Unused.
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
    """Lowercase `text`, fold the letter yo onto ye, and drop punctuation.

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
        train: bool,
    ) -> None:
        """Decode one batch and accumulate its ROUGE scores.

        Args:
            _inputs: Unused.
            targets: Batch target token ids, shape (batch, seq).
            _extras: Unused.
            predictions: Logits of shape (batch, seq, vocab).
            train: Whether this batch was part of a training pass; skipped
                unless `on_train` is set.

        Note:
            Predictions come from a teacher-forced pass, so the scores read
            higher than the ones free-running generation would produce.
        """
        if train and not self.on_train:
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
