"""Transformer encoder and decoder blocks."""

import torch
import torch.nn.functional as F
from torch import nn

from dl_roadmap.layers.attention import MultiHeadAttention


class FFN(nn.Module):
    """Position-wise feed-forward network with a GELU bottleneck."""

    def __init__(self, model_dim: int, hidden_dim: int | None = None) -> None:
        """Initializes the expand/project projection layers.

        Args:
            model_dim: Size of the input/output embeddings.
            hidden_dim: Size of the intermediate hidden representation.
                Defaults to ``4 * model_dim``.
        """
        super().__init__()

        hidden_dim = hidden_dim or 4 * model_dim

        self.expand = nn.Linear(model_dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the expand-GELU-project transformation.

        Args:
            x: Tensor of shape ``batch_size x seq_len x model_dim``.

        Returns:
            The transformed tensor of the same shape as ``x``.
        """
        hidden = self.expand(x)
        hidden = F.gelu(hidden)
        return self.project(hidden)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder block: self-attention + FFN."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the self-attention, FFN, and normalization sub-layers.

        Args:
            model_dim: Size of the input/output embeddings.
            num_heads: Number of self-attention heads.
            ffn_dim: Hidden size of the feed-forward network. Defaults to
                ``4 * model_dim`` inside ``FFN``.
            dropout: Dropout probability applied after each sub-layer.
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)

        self.ffn = FFN(model_dim, ffn_dim)

        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies pre-norm self-attention followed by a pre-norm FFN.

        Args:
            x: Tensor of shape ``batch_size x seq_len x model_dim``.
            key_padding_mask: Bool mask of shape ``batch_size x seq_len``,
                ``True`` at padded source positions.

        Returns:
            The updated hidden states, same shape as ``x``.
        """
        # Self-attention
        hidden = self.norm_1(x)
        self_output, _ = self.self_attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
        )

        x = x + self.dropout(self_output)

        # FFN
        hidden = self.norm_2(x)
        ffn_output = self.ffn(hidden)

        return x + self.dropout(ffn_output)


class TransformerDecoderLayer(nn.Module):
    """Pre-norm Transformer decoder block: self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the self-/cross-attention, FFN, and normalization sub-layers.

        Args:
            model_dim: Size of the input/output embeddings.
            num_heads: Number of attention heads (shared by both attentions).
            ffn_dim: Hidden size of the feed-forward network. Defaults to
                ``4 * model_dim`` inside ``FFN``.
            dropout: Dropout probability applied after each sub-layer.
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(model_dim, num_heads, dropout)

        self.ffn = FFN(model_dim, ffn_dim)

        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)
        self.norm_3 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies masked self-attention, cross-attention, then a pre-norm FFN.

        Args:
            x: Decoder hidden states of shape
                ``batch_size x tgt_len x model_dim``.
            memory: Encoder output of shape
                ``batch_size x src_len x model_dim``.
            tgt_key_padding_mask: Bool mask of shape
                ``batch_size x tgt_len`` for padded target positions.
            memory_key_padding_mask: Bool mask of shape
                ``batch_size x src_len`` for padded source positions.

        Returns:
            The updated hidden states, same shape as ``x``.
        """
        # Self-attention
        hidden = self.norm_1(x)
        self_output, _ = self.self_attention(
            hidden,
            hidden,
            hidden,
            causal=True,
            key_padding_mask=tgt_key_padding_mask,
        )

        x = x + self.dropout(self_output)

        # Cross-attention
        hidden = self.norm_2(x)
        cross_output, _ = self.cross_attention(
            hidden,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
        )

        x = x + self.dropout(cross_output)

        # FFN
        hidden = self.norm_3(x)
        ffn_output = self.ffn(hidden)

        return x + self.dropout(ffn_output)


class TransformerEncoder(nn.Module):
    """Stack of N pre-norm Transformer encoder blocks with a final norm."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the stack of encoder blocks and the final norm.

        Args:
            model_dim: Size of the input/output embeddings.
            num_heads: Number of self-attention heads per block.
            num_layers: Number of stacked encoder blocks.
            ffn_dim: Hidden size of each block's feed-forward network.
                Defaults to ``4 * model_dim`` inside ``FFN``.
            dropout: Dropout probability passed to every block.
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Runs the input through every encoder block and a final norm.

        Args:
            x: Tensor of shape ``batch_size x src_len x model_dim``.
            key_padding_mask: Bool mask of shape ``batch_size x src_len``,
                ``True`` at padded source positions.

        Returns:
            The encoder memory, same shape as ``x``.
        """
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Stack of N pre-norm Transformer decoder blocks with a final norm."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the stack of decoder blocks and the final norm.

        Args:
            model_dim: Size of the input/output embeddings.
            num_heads: Number of attention heads per block.
            num_layers: Number of stacked decoder blocks.
            ffn_dim: Hidden size of each block's feed-forward network.
                Defaults to ``4 * model_dim`` inside ``FFN``.
            dropout: Dropout probability passed to every block.
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(model_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Runs the target through every decoder block and a final norm.

        Args:
            x: Decoder input of shape ``batch_size x tgt_len x model_dim``.
            memory: Encoder output of shape
                ``batch_size x src_len x model_dim``.
            tgt_key_padding_mask: Bool mask of shape
                ``batch_size x tgt_len`` for padded target positions.
            memory_key_padding_mask: Bool mask of shape
                ``batch_size x src_len`` for padded source positions.

        Returns:
            The decoder output, same shape as ``x``.
        """
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(x)
