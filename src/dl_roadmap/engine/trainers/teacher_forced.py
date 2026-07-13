"""Trainer variant for seq2seq models trained with teacher forcing."""

import torch

from dl_roadmap.engine.trainer import Trainer


class TeacherForcedTrainer(Trainer):
    """Trainer that feeds ground-truth targets to the model during training.

    Passes ``targets`` as a second argument to the model's ``forward`` only
    while training, so a seq2seq decoder can use teacher forcing. During
    validation the model is called with ``None`` instead, so ``val_loss``
    reflects free-running (autoregressive) decoding quality.
    """

    def _forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, train: bool
    ) -> torch.Tensor:
        """Calls the model with targets attached only during training.

        Args:
            inputs: Batch inputs, already moved to `self.device`.
            targets: Batch targets, already moved to `self.device`.
            train: Whether this call happens during a training pass.

        Returns:
            Model predictions, passed to `self.loss_fn` alongside targets.
        """
        return self.model(inputs, targets if train else None)
