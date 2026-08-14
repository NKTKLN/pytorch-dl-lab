"""Positional encodings for sequence models."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to input embeddings."""

    pe_matrix: torch.Tensor

    def __init__(self, model_dim: int, max_length: int) -> None:
        """Precomputes the sinusoidal positional encoding table.

        Args:
            model_dim: Size of the embeddings the encoding is added to.
            max_length: Maximum sequence length supported.
        """
        super().__init__()

        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_dim)
        )

        pe_matrix = torch.zeros(max_length, model_dim, dtype=torch.float32)
        pe_matrix[:, 0::2] = torch.sin(position * frequencies)
        pe_matrix[:, 1::2] = torch.cos(
            position * frequencies[: pe_matrix[:, 1::2].shape[1]]
        )
        pe_matrix = pe_matrix.unsqueeze(0)

        self.register_buffer("pe_matrix", pe_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encodings to the input embeddings.

        Args:
            x: Tensor of shape ``batch_size x seq_len x model_dim``.

        Returns:
            The input with positional encodings added, same shape as ``x``.

        Raises:
            ValueError: If the sequence length of ``x`` exceeds ``max_length``.
        """
        sequence_length = x.shape[1]

        if sequence_length > self.pe_matrix.shape[1]:
            raise ValueError(
                f"sequence length ({sequence_length}) exceeds max_length "
                f"({self.pe_matrix.shape[1]})"
            )

        positional_encoding = self.pe_matrix[:, :sequence_length, :].to(dtype=x.dtype)

        return x + positional_encoding
