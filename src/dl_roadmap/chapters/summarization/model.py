"""Chapter 10 — transformer encoder-decoder for abstractive summarization."""

import math

import sentencepiece as spm
import torch
from torch import nn

from dl_roadmap.engine.beam_search import beam_search
from dl_roadmap.layers.positional import PositionalEncoding
from dl_roadmap.layers.transformer import TransformerDecoder, TransformerEncoder


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
        use_weight_tying: bool = True,
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
            use_weight_tying: Whether the output projection shares its weight
                matrix with the token embedding, saving ``vocab_size *
                model_dim`` parameters.
        """
        super().__init__()

        self.pad_id = sp.pad_id()
        self.bos_id = sp.bos_id()
        self.eos_id = sp.eos_id()
        self.model_dim = model_dim
        self.use_weight_tying = use_weight_tying

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

        # Weight tying: the projection keeps its own bias either way.
        if use_weight_tying:
            self.output_projection.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Rescales the token embeddings to match the ``sqrt(model_dim)`` scaling.

        An untied output projection is drawn from the same distribution, so
        that toggling `use_weight_tying` changes the sharing and nothing else.
        """
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.model_dim**-0.5)

        if not self.use_weight_tying:
            nn.init.normal_(
                self.output_projection.weight, mean=0.0, std=self.model_dim**-0.5
            )

        with torch.no_grad():
            self.embedding.weight[self.pad_id].zero_()

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

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_length: int = 128,
        beam_width: int = 3,
        length_penalty_alpha: float = 0.6,
    ) -> torch.Tensor:
        """Decodes a summary for a single source sequence via beam search.

        Args:
            x: Source token ids of shape ``1 x src_len``, encoded the same
                way as during training (``<BOS> ... <EOS>``).
            max_length: Maximum number of tokens to generate, not counting
                the leading ``<BOS>``.
            beam_width: Number of hypotheses kept alive at each step.
            length_penalty_alpha: Strength of the length penalty applied
                when ranking completed hypotheses; 0 disables it.

        Returns:
            The generated token ids, shaped ``1 x seq_len``, with the
            leading ``<BOS>`` and the trailing ``<EOS>`` stripped, ready to
            be passed to the tokenizer's decoder.

        Raises:
            ValueError: If `x` does not have a batch size of exactly 1.
        """
        if x.ndim != 2 or x.shape[0] != 1:  # noqa: PLR2004
            raise ValueError(
                f"generate decodes one example at a time, so x must have "
                f"shape 1 x src_len, got {tuple(x.shape)}"
            )

        self.eval()

        memory, src_key_padding_mask = self.encode(x)

        def logits_fn(seq: torch.Tensor) -> torch.Tensor:
            return self.decode(
                seq, memory, memory_key_padding_mask=src_key_padding_mask
            )[:, -1, :]

        sequence = beam_search(
            step=logits_fn,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_length=max_length + 1,
            beam_width=beam_width,
            length_penalty_alpha=length_penalty_alpha,
            device=x.device,
        )

        sequence = sequence[:, 1:]

        if sequence.shape[1] > 0 and int(sequence[0, -1]) == self.eos_id:
            sequence = sequence[:, :-1]

        return sequence
