"""Trainer variant feeding the decoder its own input tensor."""

import torch

from dl_roadmap.engine.trainer import Trainer


class TeacherForcingTrainer(Trainer):
    """Trainer that feeds the decoder input in both training and validation.

    Fixes the convention that a batch's first extra tensor is the decoder
    input, so encoder-decoder models can be trained without every notebook
    respelling `_forward`.
    """

    def _forward(
        self,
        inputs: torch.Tensor,
        _targets: torch.Tensor,
        extras: list[torch.Tensor],
        _train: bool,
    ) -> torch.Tensor:
        """Calls the model with the decoder input from the batch extras.

        Args:
            inputs: Batch inputs, already moved to `self.device`.
            _targets: Batch targets, already moved to `self.device`.
            extras: Batch elements beyond inputs/targets, the first of
                which is the decoder input, already moved to `self.device`.
            _train: Whether this call happens during a training pass.

        Returns:
            Model predictions, passed to `self.loss_fn` alongside targets.
        """
        decoder_input = extras[0]
        return self.model(inputs, decoder_input)  # type: ignore[no-any-return]
