"""Holds the training history and reads and writes trainer state."""

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn

from dl_roadmap.engine.trainer.optimization import OptimizationEngine

History = dict[str, list[float]]


def _legacy_state(state: dict[str, Any]) -> dict[str, Any] | None:
    """Read optimizer state saved before it moved into OptimizationEngine.

    Args:
        state: Mapping loaded from a checkpoint file.

    Returns:
        dict[str, Any] | None: State in the current layout, or None if the
            file has no flat optimizer entry.
    """
    optimizer_state = state.get("optimizer_state_dict")
    if optimizer_state is None:
        return None

    return {
        "optimizer": optimizer_state,
        "scheduler": state.get("scheduler_state_dict"),
        "scaler": state.get("scaler_state_dict"),
    }


class TrainerStateStore:
    """Own the training history and persist it with the model state."""

    def __init__(
        self,
        model: nn.Module,
        optimization: OptimizationEngine | None = None,
        checkpoint_dir: str = "",
    ) -> None:
        """Bind the state that a checkpoint has to capture.

        Args:
            model: Model whose weights are saved and restored.
            optimization: Engine whose optimizer, scheduler and scaler state
                travel with the checkpoint. None saves model state only.
            checkpoint_dir: Directory `save_checkpoint()` writes into.
        """
        self.model = model
        self.optimization = optimization
        self.checkpoint_dir = checkpoint_dir
        self.history: History = {"train_loss": [], "val_loss": []}

    def save_checkpoint(self, epoch: int) -> Path:
        """Save the trainer state into `checkpoint_dir`.

        Args:
            epoch: Current epoch number, used to name the checkpoint file.

        Returns:
            Path: The path the checkpoint was written to.

        Raises:
            ValueError: If no `checkpoint_dir` is configured.
        """
        if not self.checkpoint_dir:
            raise ValueError("save_checkpoint requires a checkpoint_dir.")

        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"

        self.save(checkpoint_path, epoch)
        logger.debug(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Restore the trainer state from a checkpoint file.

        Args:
            checkpoint_path: Path to a file written by `save_checkpoint`.

        Returns:
            int: The epoch number the checkpoint was saved at.
        """
        epoch = self.load(checkpoint_path)
        logger.debug(f"Loaded checkpoint: {checkpoint_path}, epoch={epoch}")

        return epoch

    def save(self, path: str | Path, epoch: int = 0) -> Path:
        """Save the full trainer state to a single file.

        Args:
            path: File path to write the state to. Parent directories are
                created if they don't exist.
            epoch: Epoch to record alongside the state. Defaults to 0 for
                ad-hoc saves outside of `fit`.

        Returns:
            Path: The path the state was written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }

        if self.optimization is not None:
            state["optimization_state_dict"] = self.optimization.state_dict()

        torch.save(state, path)
        logger.debug(f"Saved trainer state: {path}")

        return path

    def load(self, path: str | Path, map_location: Any | None = None) -> int:
        """Restore the full trainer state from a file written by `save`.

        Args:
            path: File path to load the state from.
            map_location: Device the tensors are loaded onto; passed straight
                to `torch.load`.

        Returns:
            int: The epoch number recorded in the saved state.

        Raises:
            FileNotFoundError: If no file exists at `path`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Trainer state not found: {path}")

        state: dict[str, Any] = torch.load(path, map_location=map_location)

        self.model.load_state_dict(state["model_state_dict"])

        optimization_state = state.get("optimization_state_dict") or _legacy_state(
            state
        )
        if self.optimization is not None and optimization_state is not None:
            self.optimization.load_state_dict(optimization_state)

        history = state.get("history")
        if history is not None:
            self.history = history

        epoch = int(state.get("epoch", 0))
        logger.debug(f"Loaded trainer state: {path}, epoch={epoch}")

        return epoch
