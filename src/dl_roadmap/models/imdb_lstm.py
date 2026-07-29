"""LSTM-based binary sentiment classifier for IMDb reviews."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class ImdbLSTM(nn.Module):
    """LSTM-based binary sentiment classifier for IMDb reviews."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_id: int = 0,
    ) -> None:
        """Initializes the model.

        Args:
            vocab_size: Number of distinct tokens in the vocabulary.
            embedding_dim: Dimensionality of the token embeddings.
            hidden_size: Number of features in the LSTM hidden state.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability applied between stacked LSTM
                layers (ignored if ``num_layers == 1``) and to the final
                hidden state before the output layer.
            pad_id: Id of the ``<PAD>`` token, used to zero its embedding
                and to derive real sequence lengths in `forward`.
        """
        super().__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_size * 2)

        self.drop = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass.

        Args:
            x: Token id sequences, padded with ``pad_id``, shaped
                ``batch_size x seq_len``.

        Returns:
            Class logits of shape ``batch_size x 2``.
        """
        lengths = (x != self.pad_id).sum(dim=1).cpu()

        emb = self.embedding(x)
        packed = pack_padded_sequence(
            emb,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        _out, (hid, _c_n) = self.lstm(packed)

        forward_hid = hid[-2]
        backward_hid = hid[-1]

        combined_hid = torch.cat(
            (forward_hid, backward_hid),
            dim=1,
        )

        features = self.norm(combined_hid)
        features = self.drop(features)

        logits = self.fc(features)
        return logits
