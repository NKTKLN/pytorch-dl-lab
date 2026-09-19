"""Lifecycle options for the training loop."""

from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:
    """Configure the training lifecycle.

    Attributes:
        epochs: Number of training epochs; must be >= 1.
        device: Torch device or device string. None requests CUDA if available,
            otherwise CPU. Resolved by `resolve_device()`.
        checkpoint_dir: Directory for epoch checkpoints. An empty string
            disables automatic checkpoint saving.
        checkpoint_every: Save a checkpoint every N epochs; must be >= 1
            when automatic checkpoint saving is enabled.
        restore_best_weights: Restore weights from the best epoch selected
            by `early_stopping`. Has no effect without that strategy.
        show_progress: Display a progress bar during training.
    """

    epochs: int = 1
    device: torch.device | str | None = None
    checkpoint_dir: str = ""
    checkpoint_every: int = 1
    restore_best_weights: bool = False
    show_progress: bool = True

    def __post_init__(self) -> None:
        """Validate the lifecycle options.

        Raises:
            ValueError: If `epochs` is below 1, or `checkpoint_every` is below
                1 with checkpoint saving enabled.
        """
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")

        if self.checkpoint_dir != "" and self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1.")

    def resolve_device(self) -> torch.device:
        """Resolve the configured device.

        Returns:
            torch.device: Requested device, or CUDA if available, otherwise CPU.
        """
        if self.device is not None:
            return torch.device(self.device)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
