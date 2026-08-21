"""Dataset and collation for abstractive summarization."""

from collections.abc import Callable
from typing import ClassVar

import pandas as pd
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

SummarizationBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class SummarizationDataset(Dataset[SummarizationBatch]):
    """Article-summary pair dataset for abstractive summarization."""

    MAX_TEXT_LEN: ClassVar[int] = 768
    MAX_SUMMARY_LEN: ClassVar[int] = 128

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
        """Encodes both columns, truncating sources and dropping long targets."""
        self.df["text_ids"] = self.df["text"].apply(self._encode).apply(self._truncate)
        self.df["summary_ids"] = self.df["summary"].apply(self._encode)

        mask = self.df["summary_ids"].str.len() <= self.MAX_SUMMARY_LEN
        self.df = self.df[mask].reset_index(drop=True)

    @classmethod
    def _truncate(cls, ids: list[int]) -> list[int]:
        """Cuts a source sequence to `MAX_TEXT_LEN`, keeping its final token.

        Args:
            ids: BOS/EOS-wrapped source token ids.

        Returns:
            The ids unchanged, or their first `MAX_TEXT_LEN - 1` followed by
            the original trailing EOS.
        """
        if len(ids) <= cls.MAX_TEXT_LEN:
            return ids

        return [*ids[: cls.MAX_TEXT_LEN - 1], ids[-1]]

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
            A tuple of the source token ids, the target token ids
            (``summary`` without the leading ``<BOS>``), and the decoder
            input token ids (``summary`` without the trailing ``<EOS>``).
        """
        row = self.df.iloc[idx]
        input_data, output_data = row["text_ids"], row["summary_ids"]

        x = torch.tensor(input_data, dtype=torch.long)
        y = torch.tensor(output_data, dtype=torch.long)

        return x, y[1:], y[:-1]


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
            batch: A list of (source, target, decoder input) token id
                sequence triples of variable length.

        Returns:
            A tuple of the padded source, target, and decoder input token
            id tensors, all shaped ``batch_size x max_seq_len``.
        """
        inputs, targets, decoder_inputs = zip(*batch)

        return (
            pad_sequence(list(inputs), batch_first=True, padding_value=pad_id),
            pad_sequence(list(targets), batch_first=True, padding_value=pad_id),
            pad_sequence(list(decoder_inputs), batch_first=True, padding_value=pad_id),
        )

    return collate_fn
