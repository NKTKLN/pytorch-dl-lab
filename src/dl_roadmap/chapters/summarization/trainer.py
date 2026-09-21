"""Trainer reading the summarization batch by field name."""

import torch

from dl_roadmap.chapters.summarization.dataset import SummarizationBatch
from dl_roadmap.engine import BaseTrainer, StepContext


class SummarizationTrainer(BaseTrainer[SummarizationBatch]):
    """Teacher-forced trainer for the encoder-decoder summarizer."""

    def _forward(self, batch: SummarizationBatch, _ctx: StepContext) -> torch.Tensor:
        """Encode the article and decode its summary one step behind.

        Args:
            batch: Source, target and decoder input, already on `self.device`.
            _ctx: Phase, epoch and step the batch belongs to; unused here.

        Returns:
            torch.Tensor: Logits over the vocabulary for every target position.
        """
        return self.model(batch.source, batch.decoder_input)  # type: ignore[no-any-return]
