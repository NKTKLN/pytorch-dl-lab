"""Transformer encoder-decoder model for abstractive summarization."""

import math

import sentencepiece as spm
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
                ``True`` positions from attention; float masks are added to
                the raw scores.
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


class Summarizer(nn.Module):
    """Transformer encoder-decoder model for abstractive summarization."""

    def __init__(  # noqa: PLR0913
        self,
        sp: spm.SentencePieceProcessor,
        model_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        max_length: int = 1024,
    ) -> None:
        """Initializes the embedding, positional encoding, and encoder/decoder stacks.

        Args:
            sp: Shared SentencePiece model for the source text and target
                summary vocabularies.
            model_dim: Size of the token embeddings and hidden states.
            num_heads: Number of attention heads per encoder/decoder block.
            num_encoder_layers: Number of stacked encoder blocks.
            num_decoder_layers: Number of stacked decoder blocks.
            ffn_dim: Hidden size of each block's feed-forward network.
                Defaults to ``4 * model_dim`` inside ``FFN``.
            dropout: Dropout probability passed to every block.
            max_length: Maximum sequence length supported by the positional encoding.
        """
        super().__init__()

        self.pad_id = sp.pad_id()
        self.bos_id = sp.bos_id()
        self.model_dim = model_dim

        self.embedding = nn.Embedding(
            num_embeddings=sp.get_piece_size(),
            embedding_dim=model_dim,
            padding_idx=self.pad_id,
        )
        self.positional_encoding = PositionalEncoding(
            model_dim=model_dim,
            max_length=max_length,
        )

        self.encoder = TransformerEncoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(model_dim, sp.get_piece_size())

        # Weight tying
        self.output_projection.weight = self.embedding.weight

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds token ids, adds positional encoding, and applies dropout.

        Args:
            x: Token ids of shape ``batch_size x seq_len``.

        Returns:
            The embedded, positionally encoded tensor, shaped
            ``batch_size x seq_len x model_dim``.
        """
        x = self.embedding(x) * math.sqrt(self.model_dim)
        x = self.positional_encoding(x)
        return self.dropout(x)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the source sequence into decoder memory.

        Args:
            x: Source token ids of shape ``batch_size x src_len``.

        Returns:
            A tuple of the encoder memory, shaped
            ``batch_size x src_len x model_dim``, and the source padding
            mask, shaped ``batch_size x src_len``, which the decoder needs
            as its ``memory_key_padding_mask``.
        """
        src_key_padding_mask = x == self.pad_id
        memory = self.encoder(self._embed(x), key_padding_mask=src_key_padding_mask)

        return memory, src_key_padding_mask

    def decode(
        self,
        decoder_input: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decodes the target sequence against precomputed encoder memory.

        No shifting happens inside this method: ``decoder_input`` is fed to
        the decoder as-is, so the caller must already supply the
        BOS-prefixed, EOS-dropped summary for teacher forcing, not the raw
        ground-truth summary.

        Args:
            decoder_input: BOS-prefixed, EOS-dropped summary token ids,
                shaped ``batch_size x tgt_len``.
            memory: Encoder output from `encode`, shaped
                ``batch_size x src_len x model_dim``.
            memory_key_padding_mask: Bool mask of shape
                ``batch_size x src_len`` for padded source positions, as
                returned by `encode`.

        Returns:
            Logits over the vocabulary, shaped
            ``batch_size x tgt_len x vocab_size``.
        """
        tgt_key_padding_mask = decoder_input == self.pad_id

        decoder_output = self.decoder(
            self._embed(decoder_input),
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.output_projection(decoder_output)

    def forward(self, x: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """Encodes the source and decodes the target sequence in parallel.

        No shifting happens inside this method: ``decoder_input`` is fed to
        the decoder as-is, so the caller must already supply the
        BOS-prefixed, EOS-dropped summary for teacher forcing, not the raw
        ground-truth summary.

        Args:
            x: Source token ids of shape ``batch_size x src_len``.
            decoder_input: BOS-prefixed, EOS-dropped summary token ids,
                shaped ``batch_size x tgt_len``.

        Returns:
            Logits over the vocabulary, shaped
            ``batch_size x tgt_len x vocab_size``.
        """
        memory, src_key_padding_mask = self.encode(x)

        return self.decode(
            decoder_input,
            memory,
            memory_key_padding_mask=src_key_padding_mask,
        )
