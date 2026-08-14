"""Character-level RNN language model."""

import torch
from torch import nn


class CharRNN(nn.Module):
    """Character-level RNN language model.

    Embeds character ids, runs them through a vanilla RNN, and projects the
    hidden states back to vocabulary logits at every timestep.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        pad_id: int = 0,
    ) -> None:
        """Initializes the model.

        Args:
            vocab_size: Number of distinct tokens in the vocabulary.
            embedding_dim: Size of the character embedding vectors.
            hidden_size: Size of the RNN hidden state.
            num_layers: Number of stacked RNN layers.
            pad_id: Token id used for padding, ignored by the embedding.
        """
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes next-character logits for a batch of sequences.

        Args:
            x: Input token ids of shape ``batch_size x seq_len``.

        Returns:
            Logits of shape ``batch_size x seq_len x vocab_size``.
        """
        emb = self.embedding(x)
        out, _hidden = self.rnn(emb)
        logits = self.fc(out)
        return logits
