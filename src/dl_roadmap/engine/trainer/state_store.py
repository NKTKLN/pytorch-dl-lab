"""Hold the training history, and read and write trainer state."""

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn

from dl_roadmap.engine.trainer.history import History, TrainingHistory
from dl_roadmap.engine.trainer.optimization import NoOptimization, Optimization


class TrainerStateStore:
    """Own the training history and persist it with the model state."""

    def __init__(
        self,
        model: nn.Module,
        optimization: Optimization | None = None,
        map_location: Any | None = None,
    ) -> None:
        """Bind the state that a checkpoint has to capture.

        Args:
            model: Model whose weights are saved and restored.
            optimization: Engine whose optimizer, learning-rate schedule and
                scaler state travel with the checkpoint. None saves model state only.
            map_location: Where `load()` puts the saved tensors; anything
                `torch.load` takes. None leaves them where they were saved.
        """
        self.model = model
        self.optimization = optimization or NoOptimization()

        self.map_location = map_location

        self.history = TrainingHistory()

    def save(self, path: str | Path, interval: int = 0) -> Path:
        """Save the full trainer state to a single file.

        Args:
            path: File path to write the state to. Parent directories are
                created if they don't exist.
            interval: Interval to record alongside the state, stored under
                "epoch" so older files still load. Defaults to 0 for ad-hoc
                saves outside of `fit`.

        Returns:
            Path: The path the state was written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "epoch": interval,
            "model_state_dict": self.model.state_dict(),
            "history": dict(self.history),
            "history_axis": list(self.history.axis),
            "history_unit": self.history.unit,
        }

        optimization_state = self.optimization.state_dict()
        if optimization_state:
            state["optimization_state_dict"] = optimization_state

        torch.save(state, path)
        logger.debug(f"Saved trainer state: {path}")

        return path

    def load(self, path: str | Path, map_location: Any | None = None) -> int:
        """Restore the full trainer state from a file written by `save`.

        Args:
            path: File path to load the state from.
            map_location: Overrides `self.map_location` for this call;
                passed straight to `torch.load`.

        Returns:
            int: The interval number recorded in the saved state; the next
                run resumes from it plus one.

        Raises:
            FileNotFoundError: If no file exists at `path`.
            KeyError: If the file carries no model state, so it was not
                written by `save`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Trainer state not found: {path}")

        if map_location is None:
            map_location = self.map_location

        state: dict[str, Any] = torch.load(path, map_location=map_location)

        if "model_state_dict" not in state:
            raise KeyError(f"Not a trainer state file: {path}")

        self.model.load_state_dict(state["model_state_dict"])

        optimization_state: dict[str, Any] | None = state.get("optimization_state_dict")
        if optimization_state is not None:
            self.optimization.load_state_dict(optimization_state)

        history: History | None = state.get("history")
        if history is not None:
            self.history = TrainingHistory(history)
            self.history.axis = list(
                state.get("history_axis")
                or range(1, len(history.get("train_loss", [])) + 1)
            )
            self.history.unit = state.get("history_unit", "epoch")

        interval = int(state.get("epoch", 0))
        logger.debug(f"Loaded trainer state: {path}, interval={interval}")

        return interval
