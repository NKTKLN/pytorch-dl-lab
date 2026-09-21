"""Dataset and collation for abstractive summarization."""

from collections.abc import Callable
from typing import ClassVar, NamedTuple

import pandas as pd
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SummarizationBatch(NamedTuple):
    """One article with the two views of its summary the decoder needs.

    Attributes:
        source: Article token ids.
        target: Summary token ids without the leading ``<BOS>`` — what the
            decoder has to predict at each position.
        decoder_input: Summary token ids without the trailing ``<EOS>`` — what
            the decoder is fed at each position, one step behind `target`.
    """

    source: torch.Tensor
    target: torch.Tensor
    decoder_input: torch.Tensor


class SummarizationDataset(Dataset[SummarizationBatch]):
    """Article-summary pair dataset for abstractive summarization."""

    MAX_ARTICLE_LENGTH: ClassVar[int] = 1280
    MAX_SUMMARY_LENGTH: ClassVar[int] = 128

    def __init__(self, df: pd.DataFrame, sp: spm.SentencePieceProcessor) -> None:
        """Initializes the dataset.

        Args:
            df: DataFrame with raw ``text`` and ``summary`` columns.
            sp: SentencePiece model used to encode both columns.
        """
        super().__init__()

        self.df = df.copy()
        self.sp = sp

        self._prepare()

    def _prepare(self) -> None:
        """Encode columns, truncate sources, and drop long targets."""
        self.df["text_ids"] = self.df["text"].apply(self._encode).apply(self._truncate)
        self.df["summary_ids"] = self.df["summary"].apply(self._encode)

        mask = self.df["summary_ids"].str.len() <= self.MAX_SUMMARY_LENGTH
        self.df = self.df[mask].reset_index(drop=True)

    @classmethod
    def _truncate(cls, ids: list[int]) -> list[int]:
        """Truncate a source sequence while preserving its final token.

        Args:
            ids: BOS/EOS-wrapped source token ids.

        Returns:
            The ids unchanged, or their first `MAX_ARTICLE_LENGTH - 1` followed by
            the original trailing EOS.
        """
        if len(ids) <= cls.MAX_ARTICLE_LENGTH:
            return ids

        return [*ids[: cls.MAX_ARTICLE_LENGTH - 1], ids[-1]]

    def _encode(self, sentence: str) -> list[int]:
        """Encodes a raw text into BOS/EOS-wrapped subword token ids.

        Args:
            sentence: The raw text to encode.

        Returns:
            The subword token ids.
        """
        ids = self.sp.encode(sentence, out_type=int)
        return [self.sp.bos_id(), *ids, self.sp.eos_id()]

    def __len__(self) -> int:
        """Returns the number of article-summary pairs in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> SummarizationBatch:
        """Returns the source, target, and decoder input sequences for a pair.

        Args:
            idx: Index of the pair in the dataset.

        Returns:
            SummarizationBatch: The source ids, and the summary shifted by one
                so that position i of `decoder_input` predicts position i of
                `target`.
        """
        row = self.df.iloc[idx]
        input_data, output_data = row["text_ids"], row["summary_ids"]

        x = torch.tensor(input_data, dtype=torch.long)
        y = torch.tensor(output_data, dtype=torch.long)

        return SummarizationBatch(source=x, target=y[1:], decoder_input=y[:-1])


def make_collate_fn(
    pad_id: int,
) -> Callable[[list[SummarizationBatch]], SummarizationBatch]:
    """Builds a collate function padding with `pad_id`.

    Args:
        pad_id: Token id used to pad every sequence in the batch.

    Returns:
        A collate function for `DataLoader`.
    """

    def collate_fn(batch: list[SummarizationBatch]) -> SummarizationBatch:
        """Pads a batch of source/target/decoder-input token id sequences.

        Args:
            batch: Per-example batches of variable length.

        Returns:
            SummarizationBatch: The padded source, target and decoder input
                tensors, all shaped ``batch_size x max_seq_len``.
        """
        sources, targets, decoder_inputs = zip(*batch)

        return SummarizationBatch(
            source=pad_sequence(list(sources), batch_first=True, padding_value=pad_id),
            target=pad_sequence(list(targets), batch_first=True, padding_value=pad_id),
            decoder_input=pad_sequence(
                list(decoder_inputs), batch_first=True, padding_value=pad_id
            ),
        )

    return collate_fn
