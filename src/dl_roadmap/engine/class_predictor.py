"""Inference utility for classification models trained with dl_roadmap."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PredictorConfig:
    """Configuration options for ClassPredictor.

    Attributes:
        device: Torch device string (e.g. "cpu", "cuda"). None auto-selects
            CUDA if available, otherwise CPU.
        batch_size: Batch size used when iterating over input tensors.
    """

    device: torch.device | str | None = None
    batch_size: int = 64


class ClassPredictor:
    """Runs inference with a trained classification model.

    Wraps a model in eval mode and turns raw input tensors into predicted
    class indices (or labels, if a `classes` mapping is given), so
    experiment scripts don't need to repeat the no_grad/argmax/DataLoader
    boilerplate.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PredictorConfig | None = None,
        classes: dict[int, Any] | None = None,
    ) -> None:
        """Initialize the predictor.

        Args:
            model: Trained model to run inference with; moved to `config.device`.
            config: Predictor options; defaults to `PredictorConfig()`.
            classes: Optional mapping of predicted class index to a label,
                used by `predict_classes`.
        """
        self.config = config or PredictorConfig()
        self.classes = classes

        self.device = self.config.device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> NDArray[np.int64]:
        """Predict class indices for a batch of inputs.

        Args:
            x: Input tensor of shape (n_samples, ...).

        Returns:
            Array of shape (n_samples,) with the predicted class index for
            each sample, in the order they appear in `x`.
        """
        self.model.eval()
        predictions: list[torch.Tensor] = []

        predict_loader = DataLoader(
            TensorDataset(x),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        for (batch_inputs,) in predict_loader:
            inputs = batch_inputs.to(self.device)

            logits = self.model(inputs)
            prediction = logits.argmax(dim=1).cpu()

            predictions.append(prediction)

        return torch.cat(predictions).numpy()

    def predict_classes(self, x: torch.Tensor) -> list[Any]:
        """Predict class labels for a batch of inputs.

        Args:
            x: Input tensor of shape (n_samples, ...).

        Returns:
            List of labels looked up in `classes` for each predicted index,
            in the order they appear in `x`.

        Raises:
            ValueError: If no `classes` mapping was provided at init time.
        """
        if self.classes is None:
            raise ValueError(
                "predict_classes requires a `classes` mapping "
                "to be set on the predictor"
            )

        raw_predictions = self.predict(x)

        return [self.classes.get(int(pred)) for pred in raw_predictions]
