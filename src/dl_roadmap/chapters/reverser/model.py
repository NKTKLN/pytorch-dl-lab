"""GRU encoder-decoder model that reverses a character sequence."""

import torch
from torch import nn


class EncoderRNN(nn.Module):
    """GRU-based sequence encoder."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_id: int = 0,
    ) -> None:
        """Initializes the model.

        Args:
            vocab_size: Number of distinct tokens in the vocabulary.
            embedding_dim: Size of the character embedding vectors.
            hidden_size: Size of the RNN hidden state.
            num_layers: Number of stacked RNN layers.
            dropout: Dropout probability applied to the embedded input.
            pad_id: Token id used for padding, ignored by the embedding.
        """
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes a batch of input sequences.

        Args:
            x: Input token ids of shape ``batch_size x seq_len``.

        Returns:
            A tuple of the per-step hidden states, shaped
            ``batch_size x seq_len x hidden_size``, and the final hidden
            state, shaped ``num_layers x batch_size x hidden_size``.
        """
        embedding = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedding)
        return output, hidden


class DecoderRNN(nn.Module):
    """GRU-based autoregressive decoder."""

    MAX_LENGTH = 64

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        output_size: int = 64,
        bos_id: int = 1,
    ) -> None:
        """Initializes the model.

        Args:
            hidden_size: Size of the RNN hidden state.
            num_layers: Number of stacked RNN layers.
            output_size: Number of distinct tokens in the vocabulary.
            bos_id: Token id used to seed the first decoding step.
        """
        super().__init__()

        self.bos_id = bos_id

        self.embedding = nn.Embedding(output_size, hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.relu = nn.ReLU()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_hidden: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decodes an output sequence from the encoder state.

        Args:
            encoder_outputs: Encoder per-step hidden states, used only to
                infer the batch size and device.
            encoder_hidden: Encoder final hidden state, used to initialize
                the decoder's hidden state.
            target: Ground-truth token ids of shape ``batch_size x seq_len``,
                used for teacher forcing while ``seq_len < MAX_LENGTH``. If
                None, decoding is free-running (greedy) for every step.

        Returns:
            A tuple of the log-probabilities over the vocabulary for each
            decoded step, shaped ``batch_size x MAX_LENGTH x output_size``,
            and the decoder's final hidden state.
        """
        device = encoder_outputs.device
        target_length = target.size(1) if target is not None else 0

        batch_size = encoder_outputs.size(0)
        dec_input = torch.full(
            (batch_size, 1), self.bos_id, dtype=torch.long, device=device
        )
        hidden = encoder_hidden
        step_outputs: list[torch.Tensor] = []

        for index in range(self.MAX_LENGTH):
            output, hidden = self.forward_step(dec_input, hidden)
            step_outputs.append(output)

            if target is not None and index < target_length:
                # Teacher forcing: feed the ground-truth character.
                dec_input = target[:, index].unsqueeze(1)
            else:
                # Greedy decoding: feed the most likely character back in.
                _, topi = output.topk(1)
                dec_input = topi.squeeze(-1).detach()

        outputs = torch.cat(step_outputs, dim=1)
        outputs = self.log_softmax(outputs)
        return outputs, hidden

    def forward_step(
        self,
        dec_input: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs a single decoding step.

        Args:
            dec_input: Token ids for the current step, shaped
                ``batch_size x 1``.
            hidden: Decoder hidden state from the previous step.

        Returns:
            A tuple of the output logits for the current step, shaped
            ``batch_size x 1 x output_size``, and the updated hidden state.
        """
        embedding = self.embedding(dec_input)
        dec_input = self.relu(embedding)
        output, hidden = self.gru(dec_input, hidden)
        output = self.fc(output)
        return output, hidden


class ReverserSeq2Seq(nn.Module):
    """Encoder-decoder model that reverses a character sequence."""

    def __init__(self, stoi: dict[str, int]) -> None:
        """Initializes the encoder and decoder.

        Args:
            stoi: Mapping from character/token to its id, used to size the
                vocabulary and locate the ``<BOS>`` token id.
        """
        super().__init__()

        self.encoder = EncoderRNN(len(stoi))
        self.decoder = DecoderRNN(output_size=len(stoi), bos_id=stoi["<BOS>"])

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Encodes ``x`` and decodes its reversal.

        Args:
            x: Input token ids of shape ``batch_size x seq_len``.
            y: Target token ids of shape ``batch_size x seq_len``, used for
                teacher forcing. If None, decoding is free-running: each
                step's predicted character is fed back in as the next input.

        Returns:
            Log-probabilities over the vocabulary for each decoded step,
            shaped ``batch_size x MAX_LENGTH x vocab_size``.
        """
        enc_output, enc_hidden = self.encoder.forward(x)
        dec_output, _dec_hidden = self.decoder.forward(enc_output, enc_hidden, y)

        return dec_output
