"""Loss functions bundled with the trainer settings they require."""

from dataclasses import dataclass

import torch
from torch import nn

from dl_roadmap.engine.loss_tracker import LossTracker, PerTokenLossTracker
from dl_roadmap.engine.trainer import GradNormalizer, LossFn


@dataclass(frozen=True)
class LossBundle:
    """A loss function together with the trainer settings it needs.

    Attributes:
        loss_fn: Callable passed to `Trainer` as its loss.
        loss_tracker: Tracker aggregating that loss over an epoch.
        grad_normalizer: Value `TrainerConfig.grad_normalizer` must take
            for the accumulated gradient to be normalized consistently
            with `loss_fn`'s reduction.
    """

    loss_fn: LossFn
    loss_tracker: LossTracker
    grad_normalizer: GradNormalizer


def make_token_loss(pad_id: int, label_smoothing: float = 0.0) -> LossBundle:
    """Builds a token-summed cross-entropy and its trainer settings.

    The reduction, the tracker, and the gradient normalizer only work as a
    set: the loss sums over non-padding tokens, the tracker divides the
    epoch total by the token count, and the trainer divides the
    accumulated gradient by the same count. Picking them separately is how
    the pairing silently breaks, so they are returned together.

    Args:
        pad_id: Target token id excluded from both the loss and the count.
        label_smoothing: Label smoothing factor for `nn.CrossEntropyLoss`.

    Returns:
        The loss function, its tracker, and the matching normalizer.
    """
    ce = nn.CrossEntropyLoss(
        ignore_index=pad_id, label_smoothing=label_smoothing, reduction="sum"
    )

    def loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the summed cross-entropy for one batch.

        Args:
            preds: Logits of shape ``batch_size x seq_len x vocab_size``.
            targets: Target token ids of shape ``batch_size x seq_len``.

        Returns:
            The scalar cross-entropy loss, ignoring `pad_id` targets.
        """
        return ce(  # type: ignore[no-any-return]
            preds.reshape(-1, preds.size(-1)), targets.reshape(-1)
        )

    return LossBundle(
        loss_fn=loss_fn,
        loss_tracker=PerTokenLossTracker(pad_id),
        grad_normalizer="loss_weights",
    )
