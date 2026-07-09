"""Save/load helpers for trained model weights."""

from pathlib import Path

import torch
from loguru import logger
from torch import nn


def save_model(model: nn.Module, path: str | Path) -> Path:
    """Save a model's weights to disk.

    Only the `state_dict` is saved (no optimizer/epoch state); use
    `Trainer.save_checkpoint` instead if you need to resume training.

    Args:
        model: Model whose weights should be saved.
        path: File path to save the weights to. Parent directories are
            created if they don't exist.

    Returns:
        The path the weights were written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)
    logger.debug(f"Saved model weights: {path}")

    return path


def load_model(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Load weights from disk into a model, in place.

    Args:
        model: Model to load the weights into; must match the
            architecture the weights were saved from.
        path: File path to load the weights from.
        device: Device to map the loaded tensors to. None keeps them on
            their original device.

    Returns:
        The same `model` instance, with weights loaded.

    Raises:
        FileNotFoundError: If no file exists at `path`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    logger.debug(f"Loaded model weights: {path}")

    return model
