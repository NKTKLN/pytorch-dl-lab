"""Attention-based GRU encoder-decoder model for English-Russian translation."""

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    """GRU-based sequence encoder."""

    def __init__(
        self,
        input_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0,
    ) -> None:
        """Initializes the model.

        Args:
            input_size: Number of distinct tokens in the input vocabulary.
            embedding_dim: Size of the token embedding vectors.
            hidden_size: Size of the RNN hidden state.
            num_layers: Number of stacked RNN layers.
            dropout: Dropout probability applied to the embedded input.
            pad_id: Token id used for padding, ignored by the embedding.
        """
        super().__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes a batch of input sequences.

        Args:
            x: Input token ids of shape ``batch_size x seq_len``.

        Returns:
            A tuple of the per-step hidden states and the final hidden state.
        """
        lengths = (x != self.pad_id).sum(dim=1).cpu()

        embedding = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(
            embedding,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        output, hidden = self.gru(packed)
        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=x.size(1)
        )
        return output, hidden


class Attention(nn.Module):
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


class DecoderRNN(nn.Module):
    """GRU-based autoregressive decoder with attention and weight tying."""

    def __init__(
        self,
        output_size: int,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Initializes the model.

        Args:
            output_size: Number of distinct tokens in the vocabulary.
            hidden_size: Size of the RNN hidden state.
            num_layers: Number of stacked RNN layers.
            dropout: Dropout probability applied to the embedded input.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_size, hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=hidden_size**-0.5)
        self.attention = Attention(hidden_size, key_size=hidden_size * 2)

        self.gru = nn.GRU(
            input_size=hidden_size + hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

        # Weight tying
        self.fc.weight = self.embedding.weight

    def forward(
        self,
        dec_input: torch.Tensor,
        hidden: torch.Tensor,
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decodes a single step, attending over the encoder outputs.

        Args:
            dec_input: Token ids for the current step, shaped
                ``batch_size x 1``.
            hidden: Decoder hidden state from the previous step.
            enc_outputs: Encoder per-step hidden states, shaped
                ``batch_size x src_len x hidden_size``.
            src_mask: True at padded source positions, excluded from
                attention.

        Returns:
            A tuple of the step logits, the updated hidden state, and the
            attention weights.
        """
        embedding = self.dropout(self.embedding(dec_input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, enc_outputs, mask=src_mask)
        dec_input = torch.cat((embedding, context), dim=2)

        output, hidden = self.gru(dec_input, hidden)
        output = self.fc(self.dropout(output))

        return output, hidden, attn_weights


class Translator(nn.Module):
    """Attention-based encoder-decoder model for English-Russian translation."""

    MAX_LENGTH = 64

    def __init__(
        self,
        sp_en: spm.SentencePieceProcessor,
        sp_ru: spm.SentencePieceProcessor,
    ) -> None:
        """Initializes the encoder and decoder.

        Args:
            sp_en: Source-language (``en``) SentencePiece model.
            sp_ru: Target-language (``ru``) SentencePiece model.
        """
        super().__init__()

        self.enc_pad_id = sp_en.pad_id()
        self.dec_bos_id = sp_ru.bos_id()

        self.teacher_forcing_ratio = 1.0

        self.encoder = EncoderRNN(sp_en.get_piece_size(), pad_id=self.enc_pad_id)
        self.decoder = DecoderRNN(sp_ru.get_piece_size())

        enc_output_size = self.encoder.gru.hidden_size * 2
        self.bridge = nn.Linear(enc_output_size, self.decoder.gru.hidden_size)

    def _bridge_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """Adapts the encoder's final hidden state to seed the decoder.

        Args:
            encoder_hidden: Encoder final hidden state, shaped
                ``(num_layers * 2) x batch_size x encoder_hidden_size``.

        Returns:
            The decoder's initial hidden state, shaped
            ``1 x batch_size x decoder_hidden_size``.
        """
        hidden = torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=-1)
        return torch.tanh(self.bridge(hidden)).unsqueeze(0)

    def forward_full_output(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes ``x`` and decodes an output sequence, step by step.

        Args:
            x: Input token ids of shape ``batch_size x seq_len``.
            y: Ground-truth token ids for teacher forcing, shaped
                ``batch_size x tgt_len``; if None, decoding is free-running
                for ``MAX_LENGTH`` steps using ``self.teacher_forcing_ratio``.

        Returns:
            A tuple of the per-step logits, the final hidden state, and
            the per-step attention weights.
        """
        encoder_outputs, encoder_hidden = self.encoder.forward(x)
        encoder_hidden = self._bridge_hidden(encoder_hidden)

        device = encoder_outputs.device
        target_length = y.size(1) if y is not None else 0
        steps = target_length if y is not None else self.MAX_LENGTH

        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full(
            (batch_size, 1),
            self.dec_bos_id,
            dtype=torch.long,
            device=device,
        )
        hidden = encoder_hidden
        step_outputs: list[torch.Tensor] = []
        step_attentions: list[torch.Tensor] = []

        for index in range(steps):
            output, hidden, attn_weights = self.decoder.forward(
                dec_input=decoder_input,
                hidden=hidden,
                enc_outputs=encoder_outputs,
                src_mask=(x == self.enc_pad_id),
            )
            step_outputs.append(output)
            step_attentions.append(attn_weights)

            if y is not None and index < target_length:
                # Teacher forcing: feed the ground-truth token.
                tf_ratio = self.teacher_forcing_ratio
                use_tf = torch.rand(batch_size, device=device) < tf_ratio
                _, topi = output.topk(1)
                greedy_input = topi.squeeze(-1).detach()
                decoder_input = torch.where(
                    use_tf.unsqueeze(1), y[:, index].unsqueeze(1), greedy_input
                )
            else:
                # Greedy decoding: feed the most likely token back in.
                _, topi = output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        outputs = torch.cat(step_outputs, dim=1)
        attentions = torch.cat(step_attentions, dim=1)

        return outputs, hidden, attentions

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Encodes ``x`` and decodes its translation.

        Args:
            x: Input token ids of shape ``batch_size x seq_len``.
            y: Target token ids for teacher forcing; if None, decoding is
                free-running.

        Returns:
            Logits over the vocabulary for each decoded step.
        """
        dec_output, _, _ = self.forward_full_output(x, y)
        return dec_output
