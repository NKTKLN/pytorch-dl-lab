"""Loss functions bundled with the trainer settings they require."""

from dl_roadmap.losses.token import LossBundle, make_token_loss

__all__ = [
    "LossBundle",
    "make_token_loss",
]
