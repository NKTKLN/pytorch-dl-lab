"""Attention mechanisms shared across sequence models."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention with optional masking."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initializes the projection layers.

        Args:
            embedding_dim: Size of the query/key/value embeddings.
            num_heads: Number of attention heads; must evenly divide
                ``embedding_dim``.
            dropout: Dropout probability applied to the attention weights.
            bias: Whether the projection layers learn an additive bias.
        """
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Splits the embedding dimension into per-head chunks.

        Args:
            x: Tensor of shape ``batch_size x seq_len x embedding_dim``.

        Returns:
            The tensor reshaped to ``batch_size x num_heads x seq_len x head_dim``.
        """
        batch_size, seq_len, _ = x.shape

        # (B, S, E) -> (B, S, H, D) -> (B, H, S, D)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merges per-head chunks back into the embedding dimension.

        Args:
            x: Tensor of shape ``batch_size x num_heads x seq_len x head_dim``.

        Returns:
            The tensor reshaped to ``batch_size x seq_len x embedding_dim``.
        """
        batch_size, num_heads, seq_len, head_dim = x.shape

        # (B, H, S, D) -> (B, S, H, D) -> (B, S, E)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batch_size, seq_len, num_heads * head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes scaled dot-product attention across all heads.

        Args:
            query: Query tensor of shape ``batch_size x q_len x embedding_dim``.
            key: Key tensor of shape ``batch_size x k_len x embedding_dim``.
            value: Value tensor of shape ``batch_size x k_len x embedding_dim``.
            attention_mask: Mask broadcastable to
                ``batch_size x num_heads x q_len x k_len``. Bool masks exclude
                ``True`` positions; float masks are added to the raw scores.
            key_padding_mask: Bool mask of shape ``batch_size x k_len``,
                ``True`` at padded key positions to exclude from attention.
            causal: Whether to additionally apply a causal mask that prevents
                each query position from attending to later key positions.

        Returns:
            A tuple of the attention output, shaped
            ``batch_size x q_len x embedding_dim``, and the attention weights.
        """
        q = self.split_heads(self.query_projection(query))
        k = self.split_heads(self.key_projection(key))
        v = self.split_heads(self.value_projection(value))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # (B, H, Q, K)
        batch_size, _, query_length, key_length = scores.shape

        if causal:
            causal_mask = torch.triu(
                torch.ones(
                    query_length,
                    key_length,
                    dtype=torch.bool,
                    device=scores.device,
                ),
                diagonal=1,
            )
            # (Q, K) -> (1, 1, Q, K)
            scores = scores.masked_fill(causal_mask[None, None, :, :], -torch.inf)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=scores.device)

            if attention_mask.ndim == 2:  # noqa: PLR2004
                # (Q, K) -> (1, 1, Q, K)
                attention_mask = attention_mask[None, None, :, :]
            elif attention_mask.ndim == 3:  # noqa: PLR2004
                # (B, Q, K) -> (B, 1, Q, K)
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.ndim != 4:  # noqa: PLR2004
                raise ValueError(
                    "attention_mask must have 2, 3, or 4 dims, got "
                    f"{attention_mask.ndim}"
                )

            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(attention_mask, -torch.inf)
            else:
                scores = scores + attention_mask.to(dtype=scores.dtype)

        if key_padding_mask is not None:
            expected_shape = (batch_size, key_length)
            if key_padding_mask.shape != expected_shape:
                raise ValueError(
                    f"key_padding_mask must have shape {expected_shape}, got "
                    f"{tuple(key_padding_mask.shape)}"
                )

            key_padding_mask = key_padding_mask.to(
                device=scores.device, dtype=torch.bool
            )
            # (B, K) -> (B, 1, 1, K)
            key_padding_mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(key_padding_mask, -torch.inf)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = self.output_projection(self.combine_heads(weights @ v))

        return output, weights


class BahdanauAttention(nn.Module):
    """Bahdanau-style additive attention."""

    def __init__(self, hidden_size: int, key_size: int | None = None) -> None:
        """Initializes the attention layers.

        Args:
            hidden_size: Size of the query vectors and the scoring space.
            key_size: Size of the key vectors; defaults to ``hidden_size``.
        """
        super().__init__()

        self.w_1 = nn.Linear(hidden_size, hidden_size)
        self.w_2 = nn.Linear(key_size or hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes a context vector for the current decoder step.

        Args:
            query: Current decoder hidden state, shaped
                ``batch_size x 1 x hidden_size``.
            keys: Encoder per-step hidden states, shaped
                ``batch_size x seq_len x key_size``.
            mask: True at padded source positions, excluded from attention.

        Returns:
            A tuple of the context vector, shaped
            ``batch_size x 1 x key_size``, and the attention weights.
        """
        scores = self.v(torch.tanh(self.w_1(query) + self.w_2(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
